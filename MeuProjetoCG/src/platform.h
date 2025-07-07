#pragma once
#include <GL/glut.h>
struct Platform {
    float x, y, width, height;
};

extern GLuint platformTextureID; 

void loadPlatformTexture();
void renderPlatform(float x, float y, float width, float height); 