#pragma once

#include <MiniFB_enums.h>
#include <WindowData.h>

@class OSXWindow;

typedef struct {
    OSXWindow           *window;
    struct mfb_timer    *timer;
} SWindowData_OSX;
