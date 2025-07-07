#include "playerLife.h"
#include <GL/glut.h>

extern int playerHealth;
extern GLuint heartFullTextureID;
extern GLuint heartEmptyTextureID;
GLuint loadTexture(const char *filename);

void loadHeartTextures()
{
    heartFullTextureID = loadTexture("./assets/textures/heart_red.png");
    heartEmptyTextureID = loadTexture("./assets/textures/heart_black.png");
}

void renderHearts()
{
    int screenWidth = 800; // Altere para sua resolução real
    int screenHeight = 600;

    int heartSize = 32;
    int spacing = 10;

    // Salvar o estado atual de projeção
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, screenWidth, 0, screenHeight); // Coordenadas da tela

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST); // Desativa profundidade para não ser escondido
    glEnable(GL_TEXTURE_2D);

    for (int i = 0; i < 5; ++i)
    {
        GLuint texture = (i < playerHealth) ? heartFullTextureID : heartEmptyTextureID;

        glBindTexture(GL_TEXTURE_2D, texture);

        int x = screenWidth - ((i + 1) * (heartSize + spacing));
        int y = screenHeight - heartSize - 10; // 10px da borda superior

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2i(x, y);
        glTexCoord2f(1, 0);
        glVertex2i(x + heartSize, y);
        glTexCoord2f(1, 1);
        glVertex2i(x + heartSize, y + heartSize);
        glTexCoord2f(0, 1);
        glVertex2i(x, y + heartSize);
        glEnd();
    }

    glDisable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);

    // Restaurar projeção
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
}
