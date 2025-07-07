#pragma once
#include <GL/glut.h>
#include "platform.h"

struct Watermelon {
    float x, y;
    bool active;
};

extern Watermelon watermelons[3];
extern GLuint watermelonTextureID;

void loadWatermelonTexture();
void renderWatermelons();
void initWatermelons(const Platform* platforms, int platformCount); 