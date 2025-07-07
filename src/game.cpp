#include "game.h"
#include "player.h"
#include "globals.h"
#include "vilao.h"
#include "ladder.h"
#include "playerLife.h"
#include "orange.h"
#include "platform.h"
#include "watermelon.h"

#include <GL/glut.h>
#include <cmath>
#include <ctime>
#include <vector>
#include <cstdlib> 
#include <cstdio>

// Prototipação
float getRandomLadderXPosition(const std::vector<float> &existingXs, float minDistance, float lastLadderX, bool trendRight);
void checkProjectileCollisions();

// =========================
// Inicialização do cenário
// =========================
void initScenario()
{
    std::srand(static_cast<unsigned int>(time(0)));

    float platformWidth = 800.0f;
    float gapY = 100;
    float minDistance = 100.0f;
    std::vector<float> usedLadderXs;

    float lastLadderX = 400.0f; // Posição inicial arbitrária para a primeira escada (meio da tela)
    bool trendRight = true;

    // Cria plataformas de ponta a ponta
    for (int i = 0; i < PLATFORM_COUNT; ++i)
    {
        float x = 0;
        float y = 100 + i * gapY;
        platforms[i] = {x, y, platformWidth, 20};
    }

    // Cria escadas
    for (int i = 0; i < LADDER_COUNT; ++i)
    {
        Platform lower = (i == 0) ? Platform{0, 0, 0, 0} : platforms[i - 1];
        Platform upper = platforms[i];

        float ladderX;
        if (i == 0) {
            ladderX = getRandomLadderXPosition(usedLadderXs, minDistance, 0.0f, true); 
        } else {
            ladderX = getRandomLadderXPosition(usedLadderXs, minDistance, lastLadderX, trendRight);
            trendRight = !trendRight; 
        }

        usedLadderXs.push_back(ladderX); 
        lastLadderX = ladderX; 

        float ladderY = (i == 0) ? 0 : lower.y + lower.height - PLAYER_HEIGHT;
        float ladderHeight = upper.y - ladderY + PLAYER_HEIGHT;

        ladders[i] = {ladderX, ladderY, 60, ladderHeight};
    }

    // Posicionar o vilão na última plataforma
    vilaoX = 400;
    vilaoY = platforms[PLATFORM_COUNT - 1].y + platforms[PLATFORM_COUNT - 1].height;
}

float getRandomLadderXPosition(const std::vector<float> &existingXs, float minDistance, float lastLadderX, bool trendRight)
{
    int margin = 40;
    int minX = margin;
    int maxX = 800 - margin - 60; 

    float preferredZoneMin = trendRight ? lastLadderX + 80 : lastLadderX - 200; 
    float preferredZoneMax = trendRight ? lastLadderX + 250 : lastLadderX - 80; 

    if (preferredZoneMin < minX) preferredZoneMin = minX;
    if (preferredZoneMax > maxX) preferredZoneMax = maxX;
    if (preferredZoneMin > preferredZoneMax) { 
        float temp = preferredZoneMin;
        preferredZoneMin = preferredZoneMax;
        preferredZoneMax = temp;
    }

    if (preferredZoneMax - preferredZoneMin < 50) { 
        preferredZoneMin = minX;
        preferredZoneMax = maxX;
    }


    for (int attempt = 0; attempt < 200; ++attempt) // Aumente o número de tentativas
    {
        float candidateX;
        if (preferredZoneMin < preferredZoneMax) {
            candidateX = preferredZoneMin + std::rand() % (static_cast<int>(preferredZoneMax - preferredZoneMin + 1));
        } else { // Fallback se a zona preferencial é inválida (ex: min > max)
            candidateX = minX + std::rand() % (maxX - minX + 1);
        }

        if (candidateX < minX) candidateX = minX;
        if (candidateX > maxX) candidateX = maxX;


        bool tooClose = false;
        for (float x : existingXs)
        {
            if (std::fabs(candidateX - x) < minDistance)
            {
                tooClose = true;
                break;
            }
        }

        if (!tooClose)
            return candidateX;
    }

    return minX + std::rand() % (maxX - minX + 1);
}

// ============================
// Atualização da física do jogador
// ============================
void updatePlayerPhysics()
{
    playerVelocityY -= gravity;
    playerY += playerVelocityY;
    playerOnGround = false;

    for (int i = 0; i < PLATFORM_COUNT; ++i)
    {
        if (ignorePlatform)
            continue;

        Platform p = platforms[i];

        float playerBottom = playerY;
        float playerTop = playerY + PLAYER_HEIGHT;
        float playerLeft = playerX;
        float playerRight = playerX + PLAYER_WIDTH;

        float platTop = p.y + p.height;
        float platLeft = p.x;
        float platRight = p.x + p.width;

        bool horizontalOverlap = playerRight > platLeft && playerLeft < platRight;
        bool verticalCollision = playerBottom <= platTop && playerBottom >= platTop - std::fabs(playerVelocityY);

        if (horizontalOverlap && verticalCollision && playerVelocityY <= 0)
        {
            playerY = platTop;
            playerVelocityY = 0;
            playerOnGround = true;
        }
    }

    // Colisão com o chão
    if (playerY < 0)
    {
        playerY = 0;
        playerVelocityY = 0;
        playerOnGround = true;
    }
}

void checkLadderCollision()
{
    playerOnLadder = false;

    float playerCenterX = playerX + PLAYER_WIDTH / 2;
    float playerBottom = playerY;
    float playerTop = playerY + PLAYER_HEIGHT;

    for (int i = 0; i < LADDER_COUNT; ++i)
    {
        Ladder l = ladders[i];

        bool inX = playerCenterX >= l.x && playerCenterX <= l.x + l.width;
        bool inY = playerBottom <= l.y + l.height && playerTop >= l.y;

        if (inX && inY)
        {
            playerOnLadder = true;
            break;
        }
    }
}


void checkWatermelonCollision() { 
    float playerLeft = playerX;
    float playerRight = playerX + PLAYER_WIDTH;
    float playerBottom = playerY;
    float playerTop = playerY + PLAYER_HEIGHT;

    for (int i = 0; i < 3; ++i) {
        if (watermelons[i].active) { 
            float watermelonLeft = watermelons[i].x; 
            float watermelonRight = watermelons[i].x + 30; 
            float watermelonBottom = watermelons[i].y;
            float watermelonTop = watermelons[i].y + 30; 

            bool overlapX = playerRight > watermelonLeft && playerLeft < watermelonRight;
            bool overlapY = playerTop > watermelonBottom && playerBottom < watermelonTop;

            if (overlapX && overlapY) {
                watermelons[i].active = false; 
                watermelonImmunityTimer = WATERMELON_IMMUNITY_DURATION; // Ativa a imortalidade por 5 segundos

                
                invulnerabilityFrames = 60; // Inicia o efeito de piscar.

                Mix_PlayChannel(-1, hitSound, 0); // Opcional: Som de pegar item
        
            }
        }
    }
}

// ======================
// Inicialização e update
// ======================
void initGame()
{
    playerHealth = 5;
    initScenario();
    initPlayer();
    loadOrangeTexture();
    initOrange();
    loadPlayerTexture();
    loadLadderTexture();
    loadHeartTextures();
    loadSounds();
    loadPlatformTexture();
    loadWatermelonTexture(); 
    initWatermelons(platforms, PLATFORM_COUNT);
}

void updateGame()
{
    if (playerHealth <= 0 && gameState == 1)
    {
        gameState = 0;
        Mix_PlayChannel(-1, gameOverSound, 0);
    }

    if (invulnerabilityFrames > 0)
        invulnerabilityFrames--;


    if (watermelonImmunityTimer > 0) {
        watermelonImmunityTimer--;
        if (invulnerabilityFrames == 0 && watermelonImmunityTimer > 0) {
            invulnerabilityFrames = 60; 
        }
    }
    checkLadderCollision();

    if (!playerOnLadder)
        updatePlayerPhysics();

    checkProjectileCollisions();
    checkWatermelonCollision();

    updatePlayer();
    updateVilao();

    updateProjectiles();
    checkVictory();

}
// ===============
// Renderização
// ===============
void renderScenario()
{
    // Plataformas
    glEnable(GL_TEXTURE_2D); 
    glBindTexture(GL_TEXTURE_2D, platformTextureID);
    glColor3f(0.6f, 0.3f, 0.1f);
    for (int i = 0; i < PLATFORM_COUNT; i++)
    {
        Platform p = platforms[i];
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); 
        glVertex2f(p.x, p.y); 
        glTexCoord2f(1.0f, 0.0f); 
        glVertex2f(p.x + p.width, p.y); 
        glTexCoord2f(1.0f, 1.0f); 
        glVertex2f(p.x + p.width, p.y + p.height); 
        glTexCoord2f(0.0f, 1.0f); 
        glVertex2f(p.x, p.y + p.height); 
        glEnd(); 
       
    }

    // Escadas com textura
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, ladderTextureID);
    glColor3f(1.0f, 1.0f, 1.0f); // branco para mostrar textura original

    for (int i = 0; i < LADDER_COUNT; ++i)
    {
        Ladder l = ladders[i];
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(l.x, l.y);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(l.x + l.width, l.y);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(l.x + l.width, l.y + l.height);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(l.x, l.y + l.height);
        glEnd();
    }

    glDisable(GL_TEXTURE_2D);

    // Laranja
    renderOrange();

    glEnd();
}

void renderGame()
{
    renderScenario();
    renderPlayer();
    renderVilao();
    renderProjectiles();
    renderPreview();
    renderHearts();
    renderWatermelons();
}

// ===============
// Entrada de Teclado
// ===============
void handleKeyPress(unsigned char key, int x, int y)
{
    playerKeyPress(key, x, y);
}

void handleKeyRelease(unsigned char key, int x, int y)
{
    playerKeyRelease(key, x, y);
}

void checkProjectileCollisions()
{
    for (auto &proj : projectiles)
    {
        if (!proj.active)
            continue;

        float projLeft = proj.x;
        float projRight = proj.x + 40; // mesmo size usado no render
        float projBottom = proj.y;
        float projTop = proj.y + 40;

        float playerLeft = playerX;
        float playerRight = playerX + PLAYER_WIDTH;
        float playerBottom = playerY;
        float playerTop = playerY + PLAYER_HEIGHT;

        bool overlapX = playerRight > projLeft && playerLeft < projRight;
        bool overlapY = playerTop > projBottom && playerBottom < projTop;

        if (overlapX && overlapY)
        {
            if (invulnerabilityFrames == 0 && playerHealth > 0)
            {
                playerHealth--;
                // printf("Player hit! Health: %d\n", playerHealth);
                playerTakeHit();
                invulnerabilityFrames = 60; // 1 segundo invulnerável
                proj.active = false;
            }
        }
    }
}
