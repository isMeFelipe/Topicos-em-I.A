#include <GL/glut.h>
#include <SDL2/SDL_mixer.h>

void initPlayer();
void updatePlayer();
void renderPlayer();
void playerKeyPress(unsigned char key, int x, int y);
void playerKeyRelease(unsigned char key, int x, int y);
void playerSpecialPress(int key);
void loadPlayerTexture();
void playerTakeHit();
void loadSounds();
GLuint loadTexture(const char *filename);

extern float playerX, playerY;
extern float playerVelocityY;
extern float playerVelocityX;
extern bool playerOnGround;
extern bool ignorePlatform;
extern int ignoreTimer;
extern GLuint playerTexture;
extern Mix_Chunk *playerHitSound;
