#pragma once
#include <GL/glut.h>

struct Ladder
{
    float x, y;
    float width, height;
};

extern GLuint ladderTextureID;

void loadLadderTexture();
void renderLadder(float x, float y, float width, float height);
