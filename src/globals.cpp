#include "globals.h"

Platform platforms[PLATFORM_COUNT];
const float gravity = 0.5f;

Ladder ladders[LADDER_COUNT];
bool playerOnLadder = false;

int playerHealth = 5;
GLuint heartFullTextureID = 0;
GLuint heartEmptyTextureID = 0;

int invulnerabilityFrames = 0;

Mix_Chunk *hitSound = nullptr;
Mix_Music *bgMusic = nullptr;
Mix_Chunk *gameOverSound = nullptr;
Mix_Music *victoryMusic = nullptr;

int gameState = 1;
float orangeX = 0.0f;
float orangeY = 0.0f;
GLuint orangeTextureID;

int watermelonImmunityTimer = 0;