#include "platform.h"
#include "globals.h" 
#include "player.h" 

GLuint platformTextureID; 

extern GLuint loadTexture(const char *filename);

void loadPlatformTexture()
{
    platformTextureID = loadTexture("./assets/textures/platform.png");
}

void renderPlatform(float x, float y, float width, float height)
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, platformTextureID);
    glColor3f(1.0f, 1.0f, 1.0f); 

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x, y);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x + width, y);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(x + width, y + height);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(x, y + height);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}