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

### Recompensa não-homogênea por **Gradiente de Interesse** com **Máscara Temporal** (memória de curto prazo)

Este método define uma recompensa **dependente do estado e do tempo** para um Grid World com obstáculos, inspirada na ideia de **gradiente de interesse** usada em patrulhamento, mas adaptada para **busca por objetivo**: o agente é recompensado ao se mover na direção de células com maior “interesse” e é desencorajado a **revisitar regiões recentes** por meio de uma máscara temporal.

A implementação utilizada neste projeto está no ambiente `gymnasium_env/grid_world_obstacles_weight.py`, com base no ambiente descrito em `gymnasium_env/grid_world_obstacles.py`.

---

#### 1) Modelagem do ambiente

- *Grid* **2D** de dimensão $5 \times 5$.
- Posição do agente no tempo $t$: $s_t = (i_t, j_t)$.
- Objetivo (*goal*): $g = (i_g, j_g)$.
- Conjunto de obstáculos: $\mathcal{O}$.
- Ação $a_t \in \{0,1,2,3\}$ produz uma **próxima posição candidata** $s'_t = (i', j')$.
  - Se for **possível** (borda ou obstáculo), o agente **não se move**: $s_{t+1}=s_t$.
  - Se **não** for **possível**, ${s}_{t+1}={s'}_{t}$.
- O episódio termina quando $s_{t+1}=g$ (*terminated*) ou quando $t \ge$ `max_steps` (*truncated*).

---

#### 2) Matrizes de interesse

##### 2.1 Importância Absoluta (estática) — $R_{\text{abs}}$

Define-se uma matriz **fixa por episódio** $R_{\text{abs}} \in \mathbb{R}^{N\times N}$:

$$
R_{\text{abs}}(i,j)=
\begin{cases}
100.0, & (i,j)=g\\
0.0, & (i,j)\in\mathcal{O}\\
20.0, & \text{caso contrário (célula livre)}
\end{cases}
$$

Interpretação: o objetivo tem alta importância; obstáculos têm importância nula; células livres têm um valor baixo, servindo como “baseline” para favorecer exploração.

##### 2.2 Máscara Temporal (memória de curto prazo) — $W_t$

A máscara temporal $W_t \in [0,1]^{N\times N}$ representa o **quanto uma célula está "recente"** (peso baixo) ou **"esquecida"** (peso alto).

Inicialização em $t=0$:

$$
W_0(i,j)=
\begin{cases}
0.0, & (i,j)\in\mathcal{O}\\
1.0, & \text{célula livre ou objetivo}
\end{cases}
$$

##### 2.3 Importância Relativa (dinâmica) — $R_t$

A importância efetiva usada para a recompensa é a **importância relativa**:

$$
R_t = W_t \odot R_{\text{abs}}
$$

onde $\odot$ é o produto elemento a elemento.

---

#### 3) Gradiente de interesse e função de recompensa

Para uma ação que propõe ir de $s_t$ para $s'_t$, define-se o **gradiente de interesse**:

$$
\mathrm{\nabla}_R(t) = R_t(s'_t) - R_t(s_t)
$$

##### 3.1 Ação não permitida (obstáculo ou borda)

Se $s'_t$ é inválido (fora do *grid*) ou $(i',j')\in\mathcal{O}$:

$$
r_t = -C_{\text{notpermited}}
\quad\text{e}\quad
s_{t+1}=s_t
$$

No cenário implementado:

- $C_{\text{notpermited}} = 10.0$.

##### 3.2 Ação permitida (movimento permitido)

Se a ação é permitida, o reward é um **mapeamento linear** do gradiente $\mathrm{\nabla}_R(t)$ para um intervalo controlado:

$$
r_t =
\left(\frac{r_{\max}-r_{\min}}{R_{\max}-R_{\min}}\right)\left(\mathrm{\nabla}_R(t)-R_{\min}\right)+r_{\min}
$$

Parâmetros usados:

- $\delta = 0.055$
- $r_{\max} = 5.0$
- $r_{\min} = -0.5$
- $R_{\max} = 100.0$
- $R_{\min} = -100.0$

Observação: como $R_t(i,j)\in[0,100]$ (pela construção de $R_{\text{abs}}$ e $W_t\in[0,1]$), então $\mathrm{\nabla}_R(t)\in[-100,100]$ e esse mapeamento tende a produzir $r_t \in [r_{\min}, r_{\max}]$ sem necessidade de *clipping*.

##### 3.3 Chegada ao objetivo

Se a ação é permitida e $s'_t = g$, atribui-se a **mesma recompensa de uma ação permitida**, mas o episódio termina:

$$
\text{terminated} \leftarrow \text{True}
$$

---

#### 4) Atualização da máscara temporal (após movimento permitido)

Após um movimento permitido para $s_{t+1}=(i',j')$, a máscara temporal é atualizada para o próximo passo:

1) **Zerar o interesse na célula recém-visitada**:

$$
W_{t+1}(i',j') = 0.0
$$

2) **Regenerar gradualmente as demais células livres**:

$$
W_{t+1}(i,j) = \min\left(1.0,\, W_t(i,j) + \delta\right),
\quad \forall (i,j)\notin\mathcal{O},\ (i,j)\neq(i',j')
$$

3) **Obstáculos permanecem nulos**:

$$
W_{t+1}(i,j)=0.0,\quad \forall (i,j)\in\mathcal{O}
$$

Intuição: se o agente tentar “andar em círculos”, ele frequentemente retorna a células com $W \approx 0$, o que reduz $R_t$ e tende a gerar $\mathrm{\nabla}_R(t)$ negativo (ou baixo), diminuindo o reward e incentivando a saída de ciclos curtos.

---

#### Referências

1. INSTITUTE OF ELECTRICAL AND ELECTRONICS ENGINEERS (IEEE). **IEEE Xplore Digital Library**. Documento 9252944. Disponível em: <https://ieeexplore.ieee.org/document/9252944>. Acesso em: 3 abr. 2026.
2. INSTITUTE OF ELECTRICAL AND ELECTRONICS ENGINEERS (IEEE). **IEEE Xplore Digital Library**. Documento 10946186. Disponível em: <https://ieeexplore.ieee.org/document/10946186>. Acesso em: 3 abr. 2026.
