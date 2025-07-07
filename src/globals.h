#include "platform.h"
#include "ladder.h"
#include "watermelon.h"
#include <SDL2/SDL_mixer.h>

#define PLATFORM_COUNT 30
extern Platform platforms[PLATFORM_COUNT];

#define LADDER_COUNT (PLATFORM_COUNT)

extern Ladder ladders[LADDER_COUNT];
extern bool playerOnLadder;

extern const float gravity;
const float PLAYER_WIDTH = 32.0f;
const float PLAYER_HEIGHT = 32.0f;

extern float vilaoX, vilaoY;
extern float vilaoDirectionX; 

extern int invulnerabilityFrames;

extern int playerHealth; // vai de 0 a 5
extern GLuint heartFullTextureID;
extern GLuint heartEmptyTextureID;


extern Mix_Music* bgMusic;
extern Mix_Chunk* hitSound;
extern Mix_Chunk* gameOverSound;
extern Mix_Music* victoryMusic;

extern int gameState;

extern float orangeX, orangeY;
extern GLuint orangeTextureID;

extern int watermelonImmunityTimer; // Contador para a imortalidade da melancia
const int WATERMELON_IMMUNITY_DURATION = 300;