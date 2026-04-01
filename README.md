# Criando ambientes customizados usando a biblioteca Gymnasium

O objetivo deste repositório é fornecer alguns exemplos de ambientes customizados criados 
usando a biblioteca Gymnasium. 

Você pode usar este arquivo README.md como um handout para entender como implementar ambientes customizados e como utilizá-los.

## Instalação

Para começar a usar este repositório você precisa clonar o repositório e instalar as dependências necessárias. Você pode fazer isso usando os seguintes comandos depois de clonar o repositório:

```bash
python -m venv venv # para criar um ambiente virtual
source venv/bin/activate # para ativar o ambiente virtual
pip install -r requirements.txt # para instalar as dependências
```

## Primeiro exemplo: ambiente GridWorld sem renderização

O primeiro exemplo é um ambiente simples de grid world. O agente pode se mover para cima, baixo, esquerda ou direita. O objetivo do agente é chegar ao objetivo (goal) o mais rápido possível. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world.py` dentro da pasta `gymnasium_env`. 

O código deste arquivo é baseado no tutorial disponível em [https://gymnasium.farama.org/introduction/create_custom_env/](https://gymnasium.farama.org/introduction/create_custom_env/). Este código tem todos os métodos necessários para criar um ambiente: `__init__`, `reset` e `step`. Só não tem o médoto `render` que é responsável por mostrar visualmente o ambiente.  

Os arquivos listados abaixo utilizam o ambiente `GridWorldEnv`: 

* `run_grid_world_v0.py`: registra o ambiente e executa um episódio, onde o comportamento do agente é aleatório.
* `run_grid_world_v0_wrapper.py`: utiliza a mesma base de código do arquivo anterior, além disso, faz uso de um wrapper para modificar a forma como o estado é retornado pelo ambiente e tratado pelo agente. 

**Questão**: Qual é a diferença entre o estado retornado pelo ambiente e o estado retornado pelo ambiente com o uso do wrapper? O que cada variável representa?

* `train_grid_world_v0.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv`. 

**Proposta**: 

* Execute o comando:

```bash
python train_grid_world_render_v0.py train
```

* Visualize a curva de aprendizado usando o plugin do tensorboard com os dados armazenados na pasta `log`. 

* Execute diversas vezes o comando: 

```bash
python train_grid_world_render_v0.py test
```

para visualizar se o agente aprendeu a melhor política. 


## Segundo exemplo: ambiente GridWorld com renderização

O segundo exemplo é o mesmo ambiente de grid world, mas agora a implementação do ambiente tem o método `render` que mostra visualmente o ambiente. A implementação deste ambiente está no arquivo `grid_world_render.py` dentro da pasta `gymnasium_env`.

Os arquivos que utilizam o ambiente `GridWorldEnv` com renderização são:

* `run_grid_world_render_v0.py`: registra o ambiente e executa um episódio, onde o comportamento do agente é aleatório.
* `run_grid_world_render_v0_wrapper.py`: utiliza a mesma base de código do arquivo anterior, além disso, faz uso de um wrapper para modificar a forma como o estado é retornado pelo ambiente e tratado pelo agente.
* `train_grid_world_render_v0.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv` com renderização.

Este último arquivo tem um código mais completo, pois o agente é treinado para atuar em um ambiente que tem uma representação visual, o modelo treinado é salvo e depois carregado para fazer uma execução do ambiente. Os dados sobre o treinamento do agente são salvos para depois serem utilizados pelo `tensorboard`.

## Terceiro exemplo: ambiente GridWorld em 3D

O terceiro exemplo é uma extensão do ambiente de grid world para um ambiente 3D. O agente pode se mover para cima, baixo, esquerda, direita, frente e trás. O objetivo do agente é chegar ao objetivo (goal) o mais rápido possível. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world_3D.py` dentro da pasta `gymnasium_env`.

O arquivo que utiliza o ambiente `GridWorldEnv` em 3D é:
* `train_grid_world_3D.py`: faz uso do algoritmo PPO da biblioteca Stable Baselines3 para treinar um agente para atuar no ambiente `GridWorldEnv` em 3D.

Existem 3 (três) formas de uso do script `train_grid_world_3D.py`:
* `python train_grid_world_3D.py train`: treina o agente e salva o modelo treinado na pasta `data` e os logs na pasta `log`.    
* `python train_grid_world_3D.py test`: carrega o modelo treinado e executa 100 episódios, calculando o percentual de sucesso do agente, entre outras métricas.
* `python train_grid_world_3D.py run`: carrega o modelo treinado e executa um único episódio, mostrando a renderização do ambiente 3D.

Para que a renderização deste ambiente aconteça, é necessário ter a biblioteca `tkinter` instalada. No Ubuntu, você pode instalar esta biblioteca com o comando:

```bash
sudo apt-get install python3-tk
```

**Importante**: esta renderização 3D foi testada apenas no sistema operacional Ubuntu.


## Quarto exemplo: ambiente GridWorld com obstáculos

O quarto exemplo é uma extensão do ambiente de grid world para incluir obstáculos. O agente deve navegar pelo ambiente evitando os obstáculos para alcançar o objetivo. O ambiente é definido na classe `GridWorldEnv` que está no arquivo `grid_world_obstacles.py` dentro da pasta `gymnasium_env`.

Para executar o treinamento do agente no ambiente com obstáculos, execute o comando:

```bash
python train_grid_world_obstacles.py train
```

Para testar o agente treinado no ambiente com obstáculos, execute o comando:

```bash
python train_grid_world_obstacles.py test
```

Esta funcionalidade irá executar o agente treinado em 100 episódios e calcular o percentual de sucesso do agente, entre outras métricas. 

Também é possível executar o agente treinado em um único episódio, para isso execute o comando:

```bash
python train_grid_world_obstacles.py run
```

## Uso do ambiente GridWorld para problemas de Coverage Path Planning

*Coverage Path Planning* (CPP) é a tarefa de planejar um caminho que passe por todos os pontos livres de um ambiente, sem colidir com obstáculos.  Diferentemente da navegação ponto a ponto (chegar a um alvo específico), o critério de sucesso do CPP é a **cobertura total** do espaço livre.

### Função de reward original (`grid_world_obstacles.py`)

O ambiente de navegação com obstáculos utiliza uma função de reward baseada em **proximidade ao alvo**:

| Evento | Reward |
|--------|--------|
| Agente alcança o alvo | +10.0 |
| Episódio truncado (max\_steps excedido) | −10.0 |
| Qualquer outro passo | `dist_anterior − dist_atual − 0.1` |

O termo `dist_anterior − dist_atual` é *reward shaping* por distância: incentiva o agente a se aproximar do alvo a cada passo.  A penalidade de −0.1 por passo estimula eficiência.

Essa função é adequada para **navegação**, mas inadequada para CPP porque:
- Não há um único alvo — o objetivo é visitar *todas* as células livres.
- O agente não recebe nenhum sinal por explorar áreas novas.

### Nova função de reward para CPP (`gymnasium_env/grid_world_cpp.py`)

Inspirada em dois trabalhos da literatura:

- **Santos et al. (2023)** — *A Deep Reinforcement Learning Approach for the Patrolling Problem of Water Resources Through Autonomous Surface Vehicles: The Ypacarai Lake Case* — que utiliza ganho de informação como sinal de reward em tarefas de patrulha/cobertura.  
- **Kiran et al. (2021)** — *A Comprehensive Survey on Coverage Path Planning for Mobile Robots in Dynamic Environments* — que descreve formulações padrão de reward para CPP, equilibrando completude (cobrir tudo) com eficiência (minimizar o comprimento do caminho).

A nova função recompensa **ganho de cobertura**:

| Evento | Reward |
|--------|--------|
| Agente entra em célula **nova** (não visitada) | +1.0 |
| Agente revisita uma célula já coberta | 0.0 |
| Custo de passo (todo passo) | −0.02 |
| Bônus de conclusão (100 % das células cobertas) | +10.0 |
| Penalidade de truncamento (max\_steps antes da cobertura total) | −5.0 |

**Por que esta formulação?**

- O reward por célula nova implementa a ideia de *information gain* de Santos et al.: o agente é incentivado proporcionalmente à área nova que descobre.
- O custo de passo + bônus de conclusão seguem o padrão CPP do survey de Kiran et al.: o agente deve equilibrar *thoroughness* (cobrir tudo) com *efficiency* (caminho curto).
- Não há penalidade por revisitar células (além do custo de passo), o que evita punir o agente quando o retrocesso é geometricamente inevitável — por exemplo, ao sair de um corredor com entrada única.

**Mudanças no espaço de observação**

Para que o agente possa planejar uma cobertura sistemática, a observação foi estendida para incluir um **mapa de cobertura** do grid:

```
obs[0:2]   — posição (x, y) do agente
obs[2:]    — mapa de cobertura linearizado (row-major), onde:
               0 = célula livre não visitada
               1 = célula livre já visitada
               2 = obstáculo
```

Dessa forma o agente tem visibilidade completa de quais células ainda precisam ser cobertas.

### Executando o ambiente CPP com um agente aleatório

Para testar o ambiente (sem renderização gráfica):

```bash
python run_grid_world_cpp.py
```

Para testar com a janela pygame (requer display):

```bash
python run_grid_world_cpp.py render
```

O script executa um episódio em um grid 5×5 com 3 obstáculos e imprime o progresso de cobertura passo a passo.  Por ser um agente aleatório, dificilmente atingirá 100 % de cobertura, mas permite verificar que o ambiente e a função de reward estão funcionando corretamente.
