#import "OSXWindow.h"
#import "OSXView.h"
#include "WindowData_OSX.h"
#include <MiniFB_internal.h>
#include <MiniFB_enums.h>


static void
set_frame_view_window_data(NSView *frame_view, SWindowData *window_data) {
    if (frame_view == nil) {
        return;
    }

    if ([frame_view isKindOfClass:[OSXView class]]) {
        ((OSXView *) frame_view)->window_data = window_data;
    }
}

@implementation OSXWindow


- (id)initWithContentRect:(NSRect)contentRect
                styleMask:(NSWindowStyleMask)windowStyle
                  backing:(NSBackingStoreType)bufferingType
                    defer:(BOOL)deferCreation
               windowData:(SWindowData *) windowData
{
    self = [super
        initWithContentRect:contentRect
        styleMask:windowStyle
        backing:bufferingType
        defer:deferCreation];

    if (self)
    {
        [self setOpaque:YES];
        [self setBackgroundColor:[NSColor clearColor]];

        self.delegate = self;

        self->window_data = windowData;
        set_frame_view_window_data([super contentView], windowData);
    }
    return self;
}


- (void) removeWindowData {
    self->window_data = 0x0;
    set_frame_view_window_data([super contentView], 0x0);
}


- (void)dealloc
{
    [[NSNotificationCenter defaultCenter]
        removeObserver:self];
    [super dealloc];
}


- (void)setContentSize:(NSSize)newSize
{
    NSSize sizeDelta = newSize;
    NSSize childBoundsSize = [childContentView bounds].size;
    sizeDelta.width -= childBoundsSize.width;
    sizeDelta.height -= childBoundsSize.height;

    OSXView *frameView = [super contentView];
    NSSize newFrameSize = [frameView bounds].size;
    newFrameSize.width += sizeDelta.width;
    newFrameSize.height += sizeDelta.height;

    [super setContentSize:newFrameSize];
}


- (void)flagsChanged:(NSEvent *)event
{
    if(window_data == 0x0)
        return;

    uint32_t mod_keys = translate_modifiers([event modifierFlags]);
    short int key_code = g_keycodes[[event keyCode] & 0x1ff];

    window_data->mod_keys = mod_keys;

    if (key_code != KB_KEY_UNKNOWN && key_code >= 0 && key_code < 512) {
        bool is_pressed = false;

        switch (key_code) {
            case KB_KEY_CAPS_LOCK:
                is_pressed = (mod_keys & KB_MOD_CAPS_LOCK) != 0;
                break;
            case KB_KEY_NUM_LOCK:
                is_pressed = (mod_keys & KB_MOD_NUM_LOCK) != 0;
                break;
            case KB_KEY_LEFT_SHIFT:
            case KB_KEY_RIGHT_SHIFT:
                is_pressed = (mod_keys & KB_MOD_SHIFT) != 0;
                break;
            case KB_KEY_LEFT_CONTROL:
            case KB_KEY_RIGHT_CONTROL:
                is_pressed = (mod_keys & KB_MOD_CONTROL) != 0;
                break;
            case KB_KEY_LEFT_ALT:
            case KB_KEY_RIGHT_ALT:
                is_pressed = (mod_keys & KB_MOD_ALT) != 0;
                break;
            case KB_KEY_LEFT_SUPER:
            case KB_KEY_RIGHT_SUPER:
                is_pressed = (mod_keys & KB_MOD_SUPER) != 0;
                break;
            default:
                is_pressed = !window_data->key_status[key_code];
                break;
        }

        if (window_data->key_status[key_code] != is_pressed) {
            window_data->key_status[key_code] = is_pressed;
            kCall(keyboard_func, key_code, mod_keys, is_pressed);
        }
    }

    [super flagsChanged:event];
}


- (void)keyDown:(NSEvent *)event
{
    if(window_data != 0x0) {
        short int key_code = g_keycodes[[event keyCode] & 0x1ff];
        window_data->mod_keys = translate_modifiers([event modifierFlags]);
        if (key_code != KB_KEY_UNKNOWN && key_code >= 0 && key_code < 512) {
            window_data->key_status[key_code] = true;
            kCall(keyboard_func, key_code, (mfb_key_mod) window_data->mod_keys, true);
        }
    }
    [childContentView.superview interpretKeyEvents:@[event]];
}


- (void)keyUp:(NSEvent *)event
{
    if(window_data != 0x0) {
        short int key_code = g_keycodes[[event keyCode] & 0x1ff];
        window_data->mod_keys = translate_modifiers([event modifierFlags]);
        if (key_code != KB_KEY_UNKNOWN && key_code >= 0 && key_code < 512) {
            window_data->key_status[key_code] = false;
            kCall(keyboard_func, key_code, (mfb_key_mod) window_data->mod_keys, false);
        }
    }
}


- (void)setContentView:(NSView *)aView
{
    if ([childContentView isEqualTo:aView]) {
        return;
    }

    NSRect bounds = [self frame];
    bounds.origin = NSZeroPoint;

    OSXView *frameView = [super contentView];
    if (!frameView)
    {
        frameView = [[[OSXView alloc] initWithFrame:bounds] autorelease];
        frameView->window_data = self->window_data;

        [super setContentView:frameView];
    }
    else {
        set_frame_view_window_data(frameView, self->window_data);
    }

    if (childContentView)
    {
        [childContentView removeFromSuperview];
    }
    childContentView = aView;
    [childContentView setFrame:[self contentRectForFrameRect:bounds]];
    [childContentView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
    [frameView addSubview:childContentView];
}


- (NSView *)contentView
{
    return childContentView;
}

// Return the real content view (the internal frame view created via super setContentView:)
- (NSView *)rootContentView
{
    return [super contentView];
}


- (BOOL)canBecomeKeyWindow
{
    return YES;
}

- (void)windowDidBecomeKey:(NSNotification *)notification
{
    kUnused(notification);
    if(window_data != 0x0) {
        window_data->is_active = true;
        kCall(active_func, true);
    }
}

- (void)windowDidResignKey:(NSNotification *)notification
{
    kUnused(notification);
    if(window_data) {
        window_data->is_active = false;
        kCall(active_func, false);
    }
}

- (BOOL)windowShouldClose:(NSWindow *) window
{
    kUnused(window);
    bool destroy = false;
    if (!window_data) {
        destroy = true;
    }
    else {
        // Obtain a confirmation of close
        if (!window_data->close_func || window_data->close_func((struct mfb_window*)window_data)) {
            destroy = true;
        }
    }

    return destroy;
}

- (void)windowWillClose:(NSNotification *)notification {
    kUnused(notification);
    if(window_data) {
        window_data->close = true;
    }
}


- (BOOL)canBecomeMainWindow
{
    return YES;
}


- (NSRect)contentRectForFrameRect:(NSRect)windowFrame
{
    windowFrame.origin = NSZeroPoint;
    return NSInsetRect(windowFrame, 0, 0);
}


+ (NSRect)frameRectForContentRect:(NSRect)windowContentRect styleMask:(NSWindowStyleMask)windowStyle
{
    kUnused(windowStyle);
    return NSInsetRect(windowContentRect, 0, 0);
}


- (void)windowDidResize:(NSNotification *)notification {
    kUnused(notification);
    if(window_data != 0x0) {
        CGSize size = [self contentRectForFrameRect:[self frame]].size;
        uint32_t resized_width  = size.width  > 0.0 ? (uint32_t) size.width  : 0u;
        uint32_t resized_height = size.height > 0.0 ? (uint32_t) size.height : 0u;

        window_data->window_width  = resized_width;
        window_data->window_height = resized_height;
        resize_dst(window_data, resized_width, resized_height);

        kCall(resize_func, (int) resized_width, (int) resized_height);
    }
}


- (void)updateCursorRects {
    NSView *frame_view = nil;
    if (self->childContentView != nil) {
        frame_view = self->childContentView.superview;
    }
    if (frame_view == nil) {
        frame_view = [super contentView];
    }

    // Ask the window to invalidate the cursor rects for the frame view.
    // The system will call -resetCursorRects on that view, where we install
    // the proper per-window cursor.
    if (frame_view != nil) {
        [self invalidateCursorRectsForView:frame_view];
    }
}

@end
