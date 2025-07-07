#include "vilao.h"
#include "player.h"
#include "platform.h"
#include "orange.h"

#include <algorithm>
#include "stb_image.h"

// ------------------------
// VARIÁVEIS DO VILÃO
// ------------------------

float vilaoX = 400;
float vilaoY = 1100;     // Início no topo da última plataforma
int vilaoDirectionX = 1; // Direção do disparo (1 = direita, -1 = esquerda)

float projectileOffsetY = 0; // Altura do disparo em relação ao vilão
const float maxOffset = 50.0f;
const float minOffset = 5.0f;

float previewX = vilaoX; // Posição do preview visual do disparo
float previewY = vilaoY + projectileOffsetY;

GLuint vilaoTextureID; // Textura do vilão

int lastLaunchTime = 0; // Controle de cooldown
const int cooldownMs = 500;
GLuint projectileTextures[3];

GLuint vilaoTextureIDs[2];
int currentVilaoFrame = 0;
int vilaoAnimationInterval = 500; // tempo entre frames em ms
int lastVilaoFrameTime = 0;

void updateVilaoAnimation();

// ------------------------
// ESTRUTURA DO PROJÉTIL
// ------------------------

std::vector<Projectile> projectiles;

// ------------------------
// INICIALIZAÇÃO
// ------------------------

void initVilao()
{
    vilaoX = 400;
    vilaoY = 1100;
    vilaoDirectionX = 1;
    projectiles.clear();
}

// ------------------------
// ATUALIZAÇÃO DO VILÃO
// ------------------------

void updateVilao()
{
    orangeX = vilaoX + 50; // Ajusta a posição da laranja
    orangeY = vilaoY;

    updateVilaoAnimation();
}

// ------------------------
// RENDERIZAÇÃO DO VILÃO
// ------------------------

void renderVilao()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, vilaoTextureIDs[currentVilaoFrame]);
    glColor3f(1, 1, 1);

    float scale = 0.3f;
    float w = 500 * scale / 2.0f;
    float h = 500 * scale / 2.0f;

    glBegin(GL_QUADS);

    if (vilaoDirectionX == 1)
    {
        glTexCoord2f(0, 0);
        glVertex2f(vilaoX - w, vilaoY);
        glTexCoord2f(1, 0);
        glVertex2f(vilaoX + w, vilaoY);
        glTexCoord2f(1, 1);
        glVertex2f(vilaoX + w, vilaoY + 2 * h);
        glTexCoord2f(0, 1);
        glVertex2f(vilaoX - w, vilaoY + 2 * h);
    }
    else
    {
        glTexCoord2f(1, 0);
        glVertex2f(vilaoX - w, vilaoY);
        glTexCoord2f(0, 0);
        glVertex2f(vilaoX + w, vilaoY);
        glTexCoord2f(0, 1);
        glVertex2f(vilaoX + w, vilaoY + 2 * h);
        glTexCoord2f(1, 1);
        glVertex2f(vilaoX - w, vilaoY + 2 * h);
    }

    glEnd();
    glDisable(GL_TEXTURE_2D);
}

// ------------------------
// DISPARO DE PROJÉTEIS
// ------------------------

void launchProjectile()
{
    int now = glutGet(GLUT_ELAPSED_TIME);
    if (now - lastLaunchTime < cooldownMs)
        return;

    lastLaunchTime = now;

    Projectile p;
    p.x = vilaoX;
    p.y = vilaoY + projectileOffsetY;
    p.vx = vilaoDirectionX * 5.0f;
    p.vy = 0;
    p.active = true;
    p.flipX = (vilaoDirectionX == -1);

    // Sorteia textura aleatória entre garfo, faca e colher
    int index = rand() % 3;
    p.textureID = projectileTextures[index];

    projectiles.push_back(p);
}

// ------------------------
// PREVIEW DO PROJÉTIL
// ------------------------

void updatePreview()
{
    previewX = vilaoX + vilaoDirectionX * 80.0f;
    previewY = vilaoY + projectileOffsetY;
}

void renderPreview()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, projectileTextures[1]); // Textura da faca

    // Cor branca com transparência (50%)
    glColor4f(1.0f, 1.0f, 1.0f, 0.7f);

    float size = 20.0f;

    glBegin(GL_QUADS);

    if (vilaoDirectionX == 1)
    {
        // Orientação normal (não espelhada)
        glTexCoord2f(0, 0);
        glVertex2f(previewX - size, previewY - size);
        glTexCoord2f(1, 0);
        glVertex2f(previewX + size, previewY - size);
        glTexCoord2f(1, 1);
        glVertex2f(previewX + size, previewY + size);
        glTexCoord2f(0, 1);
        glVertex2f(previewX - size, previewY + size);
    }
    else
    {
        // Espelhado horizontalmente
        glTexCoord2f(1, 0);
        glVertex2f(previewX - size, previewY - size);
        glTexCoord2f(0, 0);
        glVertex2f(previewX + size, previewY - size);
        glTexCoord2f(0, 1);
        glVertex2f(previewX + size, previewY + size);
        glTexCoord2f(1, 1);
        glVertex2f(previewX - size, previewY + size);
    }

    glEnd();

    glDisable(GL_TEXTURE_2D);

    // Resetar cor para o padrão
    glColor4f(1, 1, 1, 1);
}

// ------------------------
// ATUALIZAÇÃO DOS PROJÉTEIS
// ------------------------

void updateProjectiles()
{
    for (auto &p : projectiles)
    {
        if (!p.active)
            continue;

        p.x += p.vx;

        // Rebater nas paredes e descer uma "plataforma"
        if (p.x < 0 || p.x > 800)
        {
            p.vx *= -1;
            p.flipX = !p.flipX;
            p.y -= 60;
        }

        // Remove se sair da tela
        if (p.y < 0)
            p.active = false;
    }

    // Remove projéteis inativos
    projectiles.erase(
        std::remove_if(projectiles.begin(), projectiles.end(),
                       [](const Projectile &p)
                       { return !p.active; }),
        projectiles.end());
}

// ------------------------
// RENDERIZAÇÃO DOS PROJÉTEIS
// ------------------------

void renderProjectiles()
{
    for (const auto &p : projectiles)
    {
        if (!p.active)
            continue;

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, p.textureID);
        glColor3f(1, 1, 1);

        float size = 40.0f;

        glBegin(GL_QUADS);
        if (p.flipX)
        {
            // Inverter horizontalmente
            glTexCoord2f(1, 0);
            glVertex2f(p.x, p.y);
            glTexCoord2f(0, 0);
            glVertex2f(p.x + size, p.y);
            glTexCoord2f(0, 1);
            glVertex2f(p.x + size, p.y + size);
            glTexCoord2f(1, 1);
            glVertex2f(p.x, p.y + size);
        }
        else
        {
            // Normal
            glTexCoord2f(0, 0);
            glVertex2f(p.x, p.y);
            glTexCoord2f(1, 0);
            glVertex2f(p.x + size, p.y);
            glTexCoord2f(1, 1);
            glVertex2f(p.x + size, p.y + size);
            glTexCoord2f(0, 1);
            glVertex2f(p.x, p.y + size);
        }
        glEnd();

        glDisable(GL_TEXTURE_2D);
    }
}

void loadProjectileTextures()
{
    projectileTextures[0] = loadTexture("./assets/textures/fork.png");
    projectileTextures[1] = loadTexture("./assets/textures/knife.png");
    projectileTextures[2] = loadTexture("./assets/textures/spoon.png");
}

// ------------------------
// INPUT DO VILÃO (TECLAS ESPECIAIS)
// ------------------------

void handleVilaoSpecialKeyPress(int key)
{
    switch (key)
    {
    case GLUT_KEY_LEFT:
        vilaoDirectionX = -1;
        break;

    case GLUT_KEY_RIGHT:
        vilaoDirectionX = 1;
        break;

    case GLUT_KEY_UP:
        projectileOffsetY += 5.0f;
        if (projectileOffsetY > maxOffset)
            projectileOffsetY = maxOffset;
        break;

    case GLUT_KEY_DOWN:
        projectileOffsetY -= 5.0f;
        if (projectileOffsetY < minOffset)
            projectileOffsetY = minOffset;
        break;

    case GLUT_KEY_F1:
        launchProjectile();
        break;
    }

    updatePreview();
}

// ------------------------
// CARREGAMENTO DA TEXTURA DO VILÃO
// ------------------------

void loadVilaoTextures()
{
    vilaoTextureIDs[0] = loadTexture("./assets/textures/vilao_idle1.png");
    vilaoTextureIDs[1] = loadTexture("./assets/textures/vilao_idle2.png");
}

void updateVilaoAnimation()
{
    int now = glutGet(GLUT_ELAPSED_TIME);
    if (now - lastVilaoFrameTime > vilaoAnimationInterval)
    {
        currentVilaoFrame = (currentVilaoFrame + 1) % 2;
        lastVilaoFrameTime = now;
    }
}
