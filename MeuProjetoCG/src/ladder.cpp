#include "ladder.h"
#include "globals.h"
#include "player.h"

GLuint ladderTextureID;

void loadLadderTexture()
{
    ladderTextureID = loadTexture("./assets/textures/ladder.png");
}
void renderLadder(float x, float y, float width, float height)
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, ladderTextureID);
    glColor3f(1, 1, 1);

    glBegin(GL_QUADS);

    // Repete horizontalmente 1 vez e verticalmente tantas vezes quanto a altura/500
    float repeatY = height / 500.0f;

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(x, y);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(x + width, y);

    glTexCoord2f(1.0f, repeatY);
    glVertex2f(x + width, y + height);

    glTexCoord2f(0.0f, repeatY);
    glVertex2f(x, y + height);

    glEnd();

    glDisable(GL_TEXTURE_2D);
}
