#include "watermelon.h"
#include "player.h" 
#include "globals.h" 


#include <cstdlib> 
#include <ctime>  
#include <vector>  
#include <cmath>  


Watermelon watermelons[3];
GLuint watermelonTextureID;

extern GLuint loadTexture(const char *filename);

void loadWatermelonTexture() {
    watermelonTextureID = loadTexture("./assets/textures/watermelon.png"); 
}

void renderWatermelons() {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, watermelonTextureID);
    glColor3f(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 3; ++i) {
        if (watermelons[i].active) {
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(watermelons[i].x, watermelons[i].y);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(watermelons[i].x + 30, watermelons[i].y); 
            glTexCoord2f(1.0f, 1.0f); glVertex2f(watermelons[i].x + 30, watermelons[i].y + 30);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(watermelons[i].x, watermelons[i].y + 30);
            glEnd();
        }
    }
    glDisable(GL_TEXTURE_2D);
}


float getRandomXOnPlatform(const Platform& p, float itemWidth) {
    float minX = p.x;
    float maxX = p.x + p.width - itemWidth;
    if (minX < 0) minX = 0; 
    if (maxX > 800 - itemWidth) maxX = 800 - itemWidth; 

    if (minX >= maxX) return minX; 

    return minX + (std::rand() % (static_cast<int>(maxX - minX + 1)));
}

void initWatermelons(const Platform* platforms, int platformCount) {
    std::srand(static_cast<unsigned int>(time(0))); 

    std::vector<int> usedPlatformIndices; 

    float watermelonWidth = 30.0f; 

    const int MIN_PLATFORM_GAP = 7; 

    for (int i = 0; i < 3; ++i) { 
        watermelons[i].active = true;

        int targetPlatformIndex = -1;
        int attempts = 0;

        while (targetPlatformIndex == -1 && attempts < 1000) { 
            int potentialPlatformIndex;

            
            if (i == 0) {
                potentialPlatformIndex = std::rand() % (platformCount / 2 - MIN_PLATFORM_GAP);
                if (potentialPlatformIndex < 0) potentialPlatformIndex = 0;
            } else {
                int lastUsedIndex = usedPlatformIndices.back();
                int minAllowedIndex = lastUsedIndex + MIN_PLATFORM_GAP;
                
                
                int maxAllowedIndex = platformCount - 2; 
                if (maxAllowedIndex < minAllowedIndex) {
                    maxAllowedIndex = minAllowedIndex + 5; 
                }
                
                if (minAllowedIndex >= platformCount - 1) { 
                    potentialPlatformIndex = std::rand() % (platformCount - 1 - MIN_PLATFORM_GAP); 
                    if (potentialPlatformIndex < 0) potentialPlatformIndex = 0;
                } else {
                    potentialPlatformIndex = minAllowedIndex + (std::rand() % (maxAllowedIndex - minAllowedIndex + 1));
                }
            }
            
            
            if (potentialPlatformIndex >= platformCount - 1) {
                 potentialPlatformIndex = platformCount - 2; 
            }
            if (potentialPlatformIndex < 0) potentialPlatformIndex = 0; 


            bool alreadyUsed = false;
            for (int usedIdx : usedPlatformIndices) {
                if (usedIdx == potentialPlatformIndex) {
                    alreadyUsed = true;
                    break;
                }
            }


            if (!alreadyUsed) {
                targetPlatformIndex = potentialPlatformIndex;
                usedPlatformIndices.push_back(targetPlatformIndex);
            }
            attempts++;
        }

        // Fallback: Se não conseguir encontrar uma posição aleatória após muitas tentativas,
        // coloque-a sequencialmente com o espaçamento mínimo.
        if (targetPlatformIndex == -1) {
            if (i == 0) {
                targetPlatformIndex = 0; // Primeira na primeira plataforma
            } else {
                targetPlatformIndex = usedPlatformIndices.back() + MIN_PLATFORM_GAP;
            }
            // Garantir que não exceda o limite.
            if (targetPlatformIndex >= platformCount - 1) {
                targetPlatformIndex = platformCount - 2;
            }
            usedPlatformIndices.push_back(targetPlatformIndex);
        }

        Platform p = platforms[targetPlatformIndex];
        watermelons[i].x = getRandomXOnPlatform(p, watermelonWidth);
        watermelons[i].y = p.y + p.height; // Coloca em cima da plataforma
    }
}