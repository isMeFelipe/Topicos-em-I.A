#!/bin/bash

# Compila o projeto
g++ ./src/main.cpp ./src/game.cpp ./src/player.cpp ./src/globals.cpp ./src/vilao.cpp ./src/ladder.cpp ./src/playerLife.cpp ./src/orange.cpp ./src/platform.cpp ./src/watermelon.cpp -o jogo $(sdl2-config --cflags --libs) -lSDL2_mixer -lGL -lGLU -lglut

# Verifica se compilou com sucesso
if [ $? -eq 0 ]; then
    echo "Compilação OK, executando o jogo..."
    ./jogo
else
    echo "Erro na compilação, não executando."
fi
