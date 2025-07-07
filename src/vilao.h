#include "vector"
#include <GL/glut.h>

void initVilao();         // Inicialização
void updateVilao();       // Atualização por frame
void renderVilao();       // Desenho na tela
void launchProjectile();  // Dispara objeto
void updateProjectiles(); // Atualiza todos os projéteis
void renderProjectiles(); // Desenha os projéteis
void handleVilaoSpecialKeyPress(int key);
void updatePreview();
void renderPreview();
void loadVilaoTextures();      // Carrega textura do vilão
void loadProjectileTextures(); // Carrega texturas dos projéteis

struct Projectile
{
    float x, y;
    float vx, vy;
    bool active;
    GLuint textureID;
    bool flipX;
};

extern std::vector<Projectile> projectiles;