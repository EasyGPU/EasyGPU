#include <Cocoa/Cocoa.h>
#include <unistd.h>
#include <sched.h>
#include <mach/mach_time.h>
#include <stdint.h>

#include "OSXWindow.h"
#include "OSXView.h"
#include "WindowData_OSX.h"
#include <MiniFB.h>
#include <MiniFB_internal.h>
#include <MiniFB_enums.h>

//-------------------------------------
static void init_keycodes();

//-------------------------------------
static bool
calculate_buffer_layout(unsigned width, unsigned height, uint32_t *stride_out, size_t *total_out) {
    if (width == 0 || height == 0) {
        return false;
    }
    // Check for overflow on total size
    if (width > (UINT32_MAX / 4) || (uint64_t)width * 4 > (uint64_t)SIZE_MAX / height) {
        return false;
    }
    *stride_out = width * 4;
    *total_out   = (size_t)(*stride_out) * (size_t)height;
    return true;
}

//-------------------------------------
static void
destroy_window_data(SWindowData *window_data) {
    if (window_data == NULL)
        return;

    @autoreleasepool {
        SWindowData_OSX   *window_data_specific = (SWindowData_OSX *) window_data->specific;
        if (window_data_specific != NULL) {
            OSXWindow   *window = window_data_specific->window;

            if (window != nil) {
                [window removeWindowData];
                [window setDelegate:nil];
                [window close];
                [window release];
                window_data_specific->window = nil;
            }

            mfb_timer_destroy(window_data_specific->timer);
            window_data_specific->timer = NULL;

            memset(window_data_specific, 0, sizeof(SWindowData_OSX));
            free(window_data_specific);
        }

        if (window_data->draw_buffer != NULL) {
            free(window_data->draw_buffer);
            window_data->draw_buffer = NULL;
        }

        memset(window_data, 0, sizeof(SWindowData));
        free(window_data);
    }
}

//-------------------------------------
static SWindowData *
create_window_data(unsigned width, unsigned height) {
    SWindowData *window_data;
    uint32_t buffer_stride = 0;
    size_t buffer_total_bytes = 0;

    if (!calculate_buffer_layout(width, height, &buffer_stride, &buffer_total_bytes)) {
        return NULL;
    }

    window_data = malloc(sizeof(SWindowData));
    if (window_data == NULL) {
        return NULL;
    }
    memset(window_data, 0, sizeof(SWindowData));

    SWindowData_OSX *window_data_specific = malloc(sizeof(SWindowData_OSX));
    if (window_data_specific == NULL) {
        free(window_data);
        return NULL;
    }
    memset(window_data_specific, 0, sizeof(SWindowData_OSX));

    window_data->specific = window_data_specific;

    calc_dst_factor(window_data, width, height);

    window_data->buffer_width  = width;
    window_data->buffer_height = height;
    window_data->buffer_stride = buffer_stride;
    window_data->draw_buffer   = malloc(buffer_total_bytes);
    if (!window_data->draw_buffer) {
        free(window_data_specific);
        free(window_data);
        return NULL;
    }
    memset(window_data->draw_buffer, 0, buffer_total_bytes);
    window_data->is_cursor_visible = true;

    return window_data;
}

//-------------------------------------
enum { kMaxEventsPerMode = 64 };

//-------------------------------------
static void
update_events_for_mode(NSString *mode) {
    NSUInteger processed = 0;
    NSEvent *event;

    // Keep the frame loop responsive during live resize by bounding work per call.
    while (processed < kMaxEventsPerMode &&
           (event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                       untilDate:[NSDate distantPast]
                                          inMode:mode
                                         dequeue:YES])) {
        [NSApp sendEvent:event];
        ++processed;
    }
}

//-------------------------------------
static inline void
update_events(SWindowData *window_data) {
    if (window_data != NULL) {
        window_data->mouse_wheel_x = 0.0f;
        window_data->mouse_wheel_y = 0.0f;
    }

    @autoreleasepool {
        update_events_for_mode(NSDefaultRunLoopMode);
        update_events_for_mode(NSEventTrackingRunLoopMode);
        update_events_for_mode(NSModalPanelRunLoopMode);
    }
}

//-------------------------------------
struct mfb_window *
mfb_open_ex(const char *title, unsigned width, unsigned height, unsigned flags) {
    @autoreleasepool {
        const unsigned known_flags = WF_RESIZABLE | WF_FULLSCREEN | WF_FULLSCREEN_DESKTOP | WF_BORDERLESS | WF_ALWAYS_ON_TOP;
        unsigned effective_flags = flags;
        const char *window_title_c = (title != NULL && title[0] != '\0') ? title : "minifb";

        if (width == 0 || height == 0) {
            return NULL;
        }

        if ((effective_flags & ~known_flags) != 0u) {
            // Unknown flags; silently ignore
        }

        if ((effective_flags & WF_FULLSCREEN) && (effective_flags & WF_FULLSCREEN_DESKTOP)) {
            effective_flags &= ~WF_FULLSCREEN_DESKTOP;
        }

        SWindowData *window_data = create_window_data(width, height);
        if (window_data == NULL) {
            return NULL;
        }
        SWindowData_OSX *window_data_specific = (SWindowData_OSX *) window_data->specific;

        init_keycodes();

        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect              rectangle, frameRect;
        NSWindowStyleMask   styles = 0;
        bool                request_maximized_desktop = false;

        if (effective_flags & WF_BORDERLESS) {
            styles |= NSWindowStyleMaskBorderless;
        }
        else {
            styles |= NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskTitled;
        }

        if (effective_flags & WF_RESIZABLE)
            styles |= NSWindowStyleMaskResizable;

        if (effective_flags & WF_FULLSCREEN) {
            styles = NSWindowStyleMaskFullScreen;
            NSScreen *mainScreen = [NSScreen mainScreen];
            if (mainScreen == nil) {
                window_data->window_width  = width;
                window_data->window_height = height;
            }
            else {
                NSRect screenRect = [mainScreen frame];
                window_data->window_width  = screenRect.size.width;
                window_data->window_height = screenRect.size.height;
            }
            rectangle = NSMakeRect(0, 0, window_data->window_width, window_data->window_height);
            frameRect = rectangle;
        }
        else if (effective_flags & WF_FULLSCREEN_DESKTOP) {
            request_maximized_desktop = true;
            styles |= NSWindowStyleMaskResizable;
            window_data->window_width  = width;
            window_data->window_height = height;
            rectangle = NSMakeRect(0, 0, window_data->window_width, window_data->window_height);
            frameRect = [NSWindow frameRectForContentRect:rectangle styleMask:styles];
        }
        else {
            window_data->window_width  = width;
            window_data->window_height = height;
            rectangle = NSMakeRect(0, 0, window_data->window_width, window_data->window_height);
            frameRect = [NSWindow frameRectForContentRect:rectangle styleMask:styles];
        }

        window_data_specific->window = [[OSXWindow alloc] initWithContentRect:frameRect styleMask:styles backing:NSBackingStoreBuffered defer:NO windowData:window_data];
        if (!window_data_specific->window) {
            destroy_window_data(window_data);
            return NULL;
        }
        [window_data_specific->window setReleasedWhenClosed:NO];
        if (effective_flags & WF_ALWAYS_ON_TOP) {
            [window_data_specific->window setLevel:NSFloatingWindowLevel];
        }

        NSString *window_title = [NSString stringWithUTF8String:window_title_c];
        if (window_title == nil) {
            window_title = @"minifb";
        }
        [window_data_specific->window setTitle:window_title];
        [window_data_specific->window performSelectorOnMainThread:@selector(makeKeyAndOrderFront:) withObject:nil waitUntilDone:YES];
        [window_data_specific->window setAcceptsMouseMovedEvents:YES];

        [window_data_specific->window center];
        if (request_maximized_desktop) {
            [window_data_specific->window performSelectorOnMainThread:@selector(performZoom:) withObject:nil waitUntilDone:YES];
        }
        window_data_specific->timer = mfb_timer_create();
        if (window_data_specific->timer == NULL) {
            destroy_window_data(window_data);
            return NULL;
        }

        [NSApp activateIgnoringOtherApps:YES];
        [NSApp finishLaunching];

        mfb_set_keyboard_callback((struct mfb_window *) window_data, keyboard_default);

        window_data->is_initialized = true;
        return (struct mfb_window *) window_data;
    }
}

//-------------------------------------
mfb_update_state
mfb_update_ex(struct mfb_window *window, void *buffer, unsigned width, unsigned height) {
    SWindowData *window_data = (SWindowData *) window;
    uint32_t buffer_stride = 0;
    size_t total_bytes = 0;

    if (window_data == NULL) {
        return STATE_INVALID_WINDOW;
    }

    // Early exit
    if (window_data->close) {
        destroy_window_data(window_data);
        return STATE_EXIT;
    }

    if (buffer == NULL) {
        return STATE_INVALID_BUFFER;
    }
    if (!calculate_buffer_layout(width, height, &buffer_stride, &total_bytes)) {
        return STATE_INVALID_BUFFER;
    }

    SWindowData_OSX *window_data_specific = (SWindowData_OSX *) window_data->specific;
    if (window_data_specific ==  NULL) {
        return STATE_INVALID_WINDOW;
    }
    if (window_data_specific->window == nil) {
        return STATE_INVALID_WINDOW;
    }

    if (window_data->buffer_width != width || window_data->buffer_height != height) {
        void *new_draw_buffer = malloc(total_bytes);
        if (new_draw_buffer == NULL) {
            return STATE_INTERNAL_ERROR;
        }

        free(window_data->draw_buffer);
        window_data->draw_buffer = new_draw_buffer;

        window_data->buffer_width  = width;
        window_data->buffer_stride = buffer_stride;
        window_data->buffer_height = height;
    }

    if (window_data->draw_buffer == NULL) {
        return STATE_INTERNAL_ERROR;
    }
    // Copy user buffer to internal draw buffer (CGDataProvider reads asynchronously from drawRect:)
    memcpy(window_data->draw_buffer, buffer, total_bytes);

    update_events(window_data);
    if (window_data->close) {
        destroy_window_data(window_data);
        return STATE_EXIT;
    }

    // Signal the OSXView that it should redraw via drawRect:
    NSView *root_view = [window_data_specific->window rootContentView];
    if (root_view != nil) {
        [root_view setNeedsDisplay:YES];
    }

    return STATE_OK;
}

//-------------------------------------
mfb_update_state
mfb_update_events(struct mfb_window *window) {
    SWindowData *window_data = (SWindowData *) window;
    if (window_data == NULL) {
        return STATE_INVALID_WINDOW;
    }
    if (window_data->close) {
        destroy_window_data(window_data);
        return STATE_EXIT;
    }

    update_events(window_data);
    if (window_data->close) {
        destroy_window_data(window_data);
        return STATE_EXIT;
    }

    SWindowData_OSX *window_data_specific = (SWindowData_OSX *) window_data->specific;
    if (window_data_specific == NULL || window_data_specific->window == nil) {
        return STATE_INVALID_WINDOW;
    }

    // Signal the OSXView that it should redraw via drawRect:
    NSView *root_view = [window_data_specific->window rootContentView];
    if (root_view != nil) {
        [root_view setNeedsDisplay:YES];
    }

    return STATE_OK;
}

//-------------------------------------
extern double   g_time_for_frame;
extern bool     g_use_hardware_sync;

//-------------------------------------
bool
mfb_wait_sync(struct mfb_window *window) {
    SWindowData *window_data = (SWindowData *) window;
    if (window_data == NULL) {
        return false;
    }
    if (window_data->close) {
        destroy_window_data(window_data);
        return false;
    }

    SWindowData_OSX *window_data_specific = (SWindowData_OSX *) window_data->specific;
    if (window_data_specific == NULL) {
        return false;
    }
    if (window_data_specific->timer == NULL) {
        return false;
    }

    update_events(window_data);
    if (window_data->close) {
        destroy_window_data(window_data);
        return false;
    }

    // Hardware sync: no software pacing
    if (g_use_hardware_sync) {
        return true;
    }

    @autoreleasepool {
        // Software pacing: wait only the remaining time; wake on input
        for (;;) {
            double elapsed_time = mfb_timer_now(window_data_specific->timer);
            if (elapsed_time >= g_time_for_frame)
                break;

            double remaining_ms = (g_time_for_frame - elapsed_time) * 1000.0;

            if (remaining_ms > 1.5) {
                // Coarse wait with event pumping via RunLoop; leave ~1 ms margin
                CFTimeInterval timeout_s = (remaining_ms - 1.0) / 1000.0;
                if (timeout_s < 0.0)
                    timeout_s = 0.0;

                CFRunLoopRunInMode(kCFRunLoopDefaultMode, timeout_s, true);
            }
            else {
                sched_yield(); // small cooperative yield
            }

            update_events(window_data);
            if (window_data->close) {
                destroy_window_data(window_data);
                return false;
            }
        }

        mfb_timer_compensated_reset(window_data_specific->timer);
        return true;
    }
}

//-------------------------------------
extern short int g_keycodes[512];

//-------------------------------------
static void
init_keycodes() {
    static bool s_initialized = false;
    if (s_initialized) {
        return;
    }
    s_initialized = true;

    for (size_t i = 0; i < 512; ++i) {
        g_keycodes[i] = KB_KEY_UNKNOWN;
    }

    g_keycodes[0x1D] = KB_KEY_0;
    g_keycodes[0x12] = KB_KEY_1;
    g_keycodes[0x13] = KB_KEY_2;
    g_keycodes[0x14] = KB_KEY_3;
    g_keycodes[0x15] = KB_KEY_4;
    g_keycodes[0x17] = KB_KEY_5;
    g_keycodes[0x16] = KB_KEY_6;
    g_keycodes[0x1A] = KB_KEY_7;
    g_keycodes[0x1C] = KB_KEY_8;
    g_keycodes[0x19] = KB_KEY_9;
    g_keycodes[0x00] = KB_KEY_A;
    g_keycodes[0x0B] = KB_KEY_B;
    g_keycodes[0x08] = KB_KEY_C;
    g_keycodes[0x02] = KB_KEY_D;
    g_keycodes[0x0E] = KB_KEY_E;
    g_keycodes[0x03] = KB_KEY_F;
    g_keycodes[0x05] = KB_KEY_G;
    g_keycodes[0x04] = KB_KEY_H;
    g_keycodes[0x22] = KB_KEY_I;
    g_keycodes[0x26] = KB_KEY_J;
    g_keycodes[0x28] = KB_KEY_K;
    g_keycodes[0x25] = KB_KEY_L;
    g_keycodes[0x2E] = KB_KEY_M;
    g_keycodes[0x2D] = KB_KEY_N;
    g_keycodes[0x1F] = KB_KEY_O;
    g_keycodes[0x23] = KB_KEY_P;
    g_keycodes[0x0C] = KB_KEY_Q;
    g_keycodes[0x0F] = KB_KEY_R;
    g_keycodes[0x01] = KB_KEY_S;
    g_keycodes[0x11] = KB_KEY_T;
    g_keycodes[0x20] = KB_KEY_U;
    g_keycodes[0x09] = KB_KEY_V;
    g_keycodes[0x0D] = KB_KEY_W;
    g_keycodes[0x07] = KB_KEY_X;
    g_keycodes[0x10] = KB_KEY_Y;
    g_keycodes[0x06] = KB_KEY_Z;

    g_keycodes[0x27] = KB_KEY_APOSTROPHE;
    g_keycodes[0x2A] = KB_KEY_BACKSLASH;
    g_keycodes[0x2B] = KB_KEY_COMMA;
    g_keycodes[0x18] = KB_KEY_EQUAL;
    g_keycodes[0x32] = KB_KEY_GRAVE_ACCENT;
    g_keycodes[0x21] = KB_KEY_LEFT_BRACKET;
    g_keycodes[0x1B] = KB_KEY_MINUS;
    g_keycodes[0x2F] = KB_KEY_PERIOD;
    g_keycodes[0x1E] = KB_KEY_RIGHT_BRACKET;
    g_keycodes[0x29] = KB_KEY_SEMICOLON;
    g_keycodes[0x2C] = KB_KEY_SLASH;
    g_keycodes[0x0A] = KB_KEY_WORLD_1;

    g_keycodes[0x33] = KB_KEY_BACKSPACE;
    g_keycodes[0x39] = KB_KEY_CAPS_LOCK;
    g_keycodes[0x75] = KB_KEY_DELETE;
    g_keycodes[0x7D] = KB_KEY_DOWN;
    g_keycodes[0x77] = KB_KEY_END;
    g_keycodes[0x24] = KB_KEY_ENTER;
    g_keycodes[0x35] = KB_KEY_ESCAPE;
    g_keycodes[0x7A] = KB_KEY_F1;
    g_keycodes[0x78] = KB_KEY_F2;
    g_keycodes[0x63] = KB_KEY_F3;
    g_keycodes[0x76] = KB_KEY_F4;
    g_keycodes[0x60] = KB_KEY_F5;
    g_keycodes[0x61] = KB_KEY_F6;
    g_keycodes[0x62] = KB_KEY_F7;
    g_keycodes[0x64] = KB_KEY_F8;
    g_keycodes[0x65] = KB_KEY_F9;
    g_keycodes[0x6D] = KB_KEY_F10;
    g_keycodes[0x67] = KB_KEY_F11;
    g_keycodes[0x6F] = KB_KEY_F12;
    g_keycodes[0x69] = KB_KEY_F13;
    g_keycodes[0x6B] = KB_KEY_F14;
    g_keycodes[0x71] = KB_KEY_F15;
    g_keycodes[0x6A] = KB_KEY_F16;
    g_keycodes[0x40] = KB_KEY_F17;
    g_keycodes[0x4F] = KB_KEY_F18;
    g_keycodes[0x50] = KB_KEY_F19;
    g_keycodes[0x5A] = KB_KEY_F20;
    g_keycodes[0x73] = KB_KEY_HOME;
    g_keycodes[0x72] = KB_KEY_INSERT;
    g_keycodes[0x7B] = KB_KEY_LEFT;
    g_keycodes[0x3A] = KB_KEY_LEFT_ALT;
    g_keycodes[0x3B] = KB_KEY_LEFT_CONTROL;
    g_keycodes[0x38] = KB_KEY_LEFT_SHIFT;
    g_keycodes[0x37] = KB_KEY_LEFT_SUPER;
    g_keycodes[0x6E] = KB_KEY_MENU;
    g_keycodes[0x47] = KB_KEY_NUM_LOCK;
    g_keycodes[0x79] = KB_KEY_PAGE_DOWN;
    g_keycodes[0x74] = KB_KEY_PAGE_UP;
    g_keycodes[0x7C] = KB_KEY_RIGHT;
    g_keycodes[0x3D] = KB_KEY_RIGHT_ALT;
    g_keycodes[0x3E] = KB_KEY_RIGHT_CONTROL;
    g_keycodes[0x3C] = KB_KEY_RIGHT_SHIFT;
    g_keycodes[0x36] = KB_KEY_RIGHT_SUPER;
    g_keycodes[0x31] = KB_KEY_SPACE;
    g_keycodes[0x30] = KB_KEY_TAB;
    g_keycodes[0x7E] = KB_KEY_UP;

    g_keycodes[0x52] = KB_KEY_KP_0;
    g_keycodes[0x53] = KB_KEY_KP_1;
    g_keycodes[0x54] = KB_KEY_KP_2;
    g_keycodes[0x55] = KB_KEY_KP_3;
    g_keycodes[0x56] = KB_KEY_KP_4;
    g_keycodes[0x57] = KB_KEY_KP_5;
    g_keycodes[0x58] = KB_KEY_KP_6;
    g_keycodes[0x59] = KB_KEY_KP_7;
    g_keycodes[0x5B] = KB_KEY_KP_8;
    g_keycodes[0x5C] = KB_KEY_KP_9;
    g_keycodes[0x45] = KB_KEY_KP_ADD;
    g_keycodes[0x41] = KB_KEY_KP_DECIMAL;
    g_keycodes[0x4B] = KB_KEY_KP_DIVIDE;
    g_keycodes[0x4C] = KB_KEY_KP_ENTER;
    g_keycodes[0x51] = KB_KEY_KP_EQUAL;
    g_keycodes[0x43] = KB_KEY_KP_MULTIPLY;
    g_keycodes[0x4E] = KB_KEY_KP_SUBTRACT;
}

//-------------------------------------
bool
mfb_set_viewport(struct mfb_window *window, unsigned offset_x, unsigned offset_y, unsigned width, unsigned height) {
    SWindowData *window_data = (SWindowData *) window;

    if (window_data == NULL) {
        return false;
    }

    if (offset_x + width > window_data->window_width) {
        return false;
    }
    if (offset_y + height > window_data->window_height) {
        return false;
    }

    window_data->dst_offset_x = offset_x;
    window_data->dst_offset_y = offset_y;
    window_data->dst_width    = width;
    window_data->dst_height   = height;
    calc_dst_factor(window_data, window_data->window_width, window_data->window_height);

    return true;
}

//-------------------------------------
void
mfb_get_monitor_scale(struct mfb_window *window, float *scale_x, float *scale_y) {
    float scale = 1.0f;

    if (window != NULL) {
        SWindowData     *window_data = (SWindowData *) window;
        SWindowData_OSX *window_data_specific = (SWindowData_OSX *) window_data->specific;
        if (window_data_specific != NULL && window_data_specific->window != nil) {
            scale = [window_data_specific->window backingScaleFactor];
        }
        else {
            scale = [[NSScreen mainScreen] backingScaleFactor];
        }
    }
    else {
        scale = [[NSScreen mainScreen] backingScaleFactor];
    }

    if (scale_x) {
        *scale_x = scale;
        if (*scale_x == 0) {
            *scale_x = 1;
        }
    }

    if (scale_y) {
        *scale_y = scale;
        if (*scale_y == 0) {
            *scale_y = 1;
        }
    }
}

//-------------------------------------
extern double   g_timer_frequency;
extern double   g_timer_resolution;

//-------------------------------------
uint64_t
mfb_timer_tick() {
    static mach_timebase_info_data_t    timebase = { 0 };

    if (timebase.denom == 0) {
        (void) mach_timebase_info(&timebase);
    }

    uint64_t time = mach_absolute_time();

    // Perform the arithmetic at 128-bit precision to avoid overflow
    uint64_t high     = (time >> 32) * timebase.numer;
    uint64_t high_rem = ((high % timebase.denom) << 32) / timebase.denom;
    uint64_t low      = (time & 0xFFFFFFFFull) * timebase.numer / timebase.denom;
    high /= timebase.denom;

    return (high << 32) + high_rem + low;
}

//-------------------------------------
void
mfb_timer_init() {
    g_timer_frequency  = 1e+9;
    g_timer_resolution = 1.0 / g_timer_frequency;
}

//-------------------------------------
void mfb_show_cursor(struct mfb_window *window, bool show) {
    SWindowData *window_data = (SWindowData *) window;
    if (window_data == NULL) {
        return;
    }

    @autoreleasepool {
        if (window_data->is_cursor_visible != show) {
            window_data->is_cursor_visible = show;

            // Update cursor rects on the window to use per-window
            // invisible cursor instead of hiding the global cursor.
            SWindowData_OSX *window_data_specific = (SWindowData_OSX *) window_data->specific;
            if (window_data_specific && window_data_specific->window) {
                [window_data_specific->window performSelectorOnMainThread:@selector(updateCursorRects) withObject:nil waitUntilDone:YES];
            }
        }
    }
}
