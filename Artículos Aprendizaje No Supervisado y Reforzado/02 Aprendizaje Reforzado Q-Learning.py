#!/usr/bin/env python
# coding: utf-8

# # Análisis de Aprendizaje Reforzado
# 
# En el presente artículo analizaremos de forma prática modelos de Aprendizaje Reformado, un paradigma de Machine Learning en el que un agente aprende una política óptima para interactuar con un entorno. Bien, a través de un proceso de prueba y error, el agente observa estados, toma acciones y recibe recompensas que refuerzan o penalizan sus decisiones. El objetivo es maximizar la recompensa acumulada a lo largo del tiempo, utilizando técnicas como programación dinámica, métodos Monte Carlo o aprendizaje Q (Q-Learning).

# <section style="width: 100%;"> 
# 	<div style="width: 50%; float: left; border-radius: 10px;"> 
#         Para ver un caso práctico de aprendizaje reforzado, usaremos la liberia de <code>gymnasium</code> que incluye diferentes juegos donde se puede usar este tipo de aprendizajes. En este caso nos centraremos en el algoritmo de <em>Q-learning</em> así que nos va bien <a href="https://gymnasium.farama.org/environments/toy_text/frozen_lake/">frozen lake</a>.<br>
# 		Este consiste en que el agente tiene que llegar al regalo para ganar la partida. <br>
# 		Tiene cuatro acciones para hacerlo:
# 		<ul>
# 			<li> 0: mover izquierda
# 			<li> 1: mover abajo
# 			<li> 2: mover derecha
# 			<li> 3: mover arriba
# 		</ul>
# 		El espacio con el que se cuenta se puede ver en la imagen a la derecha, donde tenemos una cuadricula 4x4, donde nuestro agente comienza en la casilla (0,0) y su casilla objectivo esta en (3,3). <br>
# 		La partida acaba cuando llega a la casilla objetivo o a una de lago.
# 	</div> <div style="width: 50%; float: left;">
# 		<img src="https://gymnasium.farama.org/_images/frozen_lake.gif" />
# 	 </div>
# </section>

# ¡Comenzamos!

# ## 1 Importación de librerías y paquetes

# In[99]:


import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper
import random

import warnings
warnings.filterwarnings("ignore") 


# ## 2 Inicialización del entorno y de los datos
# 
# ### 2.1 Entornos de trabajo
# 
# Inicializamos los entornos de entranmiento (env_train) y pruebas (env_test) de FrozenLake usando la biblioteca Gym. De esta manera, podremos entrenar al agente en un entorno más desafiante (con deslizamiento) y luego probarlo mientras se observa su rendimiento en tiempo real. 

# In[100]:


env_train = gym.make('FrozenLake-v1', is_slippery=True)
env_test = gym.make('FrozenLake-v1', render_mode='human')


# ### 2.2 Q-Table
# 
# A continuación, definiremos una Q-Table, también denominada tabla Q, la cual compone una estructura de datos fundamental en el aprendizaje por refuerzo, utilizada en algoritmos como el Q-Learning. Representa una matriz que almacena los valores Q, que indican la calidad o utilidad esperada de realizar una acción en un estado determinado, basándose en la recompensa acumulada futura esperada.
# 
# En una tabla Q (Q-table) cada columna representa cada una de las acciones y las filas son los estados posibles del entorno. Cada celda contiene el valor Q, que representa la calidad de la acción para llegar al objetivo.

# In[101]:


action_size = env_train.action_space.n
state_size = env_train.observation_space.n


# In[102]:


q_table = np.zeros((state_size, action_size))
print(q_table)


# ### 2.3 Rewards
# 
# Como último paso definiremos la recompensa, un valor numérico que el agente recibe después de tomar una acción en un entorno. Por lo tanto, sirve como retroalimentación para indicar qué tan buena o mala fue esa acción en función del objetivo del agente. La recompensa es un punto clave del proceso de aprendizaje del agente, ya que lo va guiando ayudándole a descubrir una política óptima (la mejor manera de actuar en cada estado).
# 
# La recompensa la modificaremos manualmente para dar peso a cuando llega al estado objetivo, cuando cae en un lago o cuando cae en una casilla neutra. Como queremos remarcar que llegue al objetivo, le daremos a este el valor más alto, al lago el más bajo ya que seria un estado donde el agente ha perdido y cada paso lo penalizaremos ligeramente para intentar evitar que de vueltas por el mapa haciendo caminos más largos.
# - Objetivo +1 punto
# - Lago -1 punto
# - Pasos -0.01 punto

# In[103]:


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)

    def reward(self, reward):
        current_state = self.env.s

        if current_state == 15:  # objetivo
            return 1
        elif current_state in [5, 7, 11, 12]:  # lago
            return -1
        else:  # pasos
            return -0.001
        
env_train = CustomRewardWrapper(env_train)
env_test = CustomRewardWrapper(env_test)


# # 3. Algoritmo Q-Learning
# 
# Aplicaremos en este artículo Q-Learning, un algoritmo de aprendizaje por refuerzo que permite a un agente aprender una política óptima mediante la actualización iterativa de una Q-Table. Este modelo evalúa el valor de cada acción en un estado específico, maximizando la recompensa acumulada a largo plazo sin requerir un modelo explícito del entorno.
# 
# Para aplicar dicho modelo, primero es necesario definir una serie de parámetros generales y de exploración, los cuales equilibran el aprendizaje y la toma de decisiones del agente, asegurando un proceso eficiente de exploración y explotación del entorno para encontrar una política óptima.
# 
# Mostramos a continuación una imagen de la formula del algoritmo Q-Learning para analizar el impacto de cada parámetro en el resultado final que se espera obtener:
# 
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a3a4d2ac903b1be02cc81e60de2e9f91d7025fec)

# In[104]:


# Parametros generales
total_episodes = 50000        # Episodios totales
learning_rate = 0.1           # Learning rate
max_steps = 99                # Pasos maximos por episodio
gamma = 0.95                  # Tasa de descuento

# Parametros de exploración
epsilon = 1.0                 # Tasa de exploración
max_epsilon = 1.0             # Probabilidad de explorar al empezar
min_epsilon = 0.01             # Probabilidad minima de explorar 
decay_rate = 0.1              # Tasa de decaimiento exponencial de la probabilidad de exploración


# Una vez definidos los parámetros generales y de exploración, procedemos a aplicar el modelo Q-Learning y proceder con su entrenamiento a través del entorno de entrenamiento (env_train).

# In[105]:


rewards = []
steps = []
finished = []

for episode in range(total_episodes):
    # Reiniciar el entorno
    state, info = env_train.reset()
    step = 0
    total_rewards = 0
    
    for step in range(max_steps):
        # Elegir una acción a en el estado actual del mundo (s)
        ## Primero, aleatorizamos un número
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## Si este número > mayor que epsilon --> explotación (tomando el valor Q más alto para este estado)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(q_table[state,:])

        # De lo contrario, hacer una elección aleatoria --> exploración
        else:
            action = env_train.action_space.sample()

        # Tomar la acción (a) y observar el estado resultante (s') y la recompensa (r)
        new_state, reward, terminated, truncated, info = env_train.step(action)

        # Actualizar Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # q_table[new_state,:] : todas las acciones que podemos tomar desde el nuevo estado
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        
        total_rewards = total_rewards + reward
        
        # Nuestro nuevo estado es state
        state = new_state
        
        # Si ha terminado (si morimos): finalizar episodio
        if truncated or terminated: 
            finished.append(reward > 0)
            break
        
    episode += 1
    steps.append(step)
    # Reducir epsilon (porque necesitamos cada vez menos exploración)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)



print("Percent of episodes finished successfully: {0}".format(sum(finished)/(total_episodes)))
print("Percent of episodes finished successfully (last 100 episodes): {0}".format(sum(finished[-100:])/(100)))
print("Average number of steps: %.2f" % (sum(steps)/(total_episodes)))
print("Average number of steps (last 100 episodes): %.2f" % (sum(steps[-100:])/(100)))
print('\n')
print(q_table)
print(epsilon)


# Tras el entrenamiento, podemos ver que en los ultimo episodeos hay una mejora del rendimiento donde el porcentaje de finalización es del 67.78% pero sin embargo de los 100 últimos, un 85% ha acabado em el objetivo final. Además, también hay una reducción media de los pasos, por lo tanto esta aprendiendo a llegar al objetivo y además reduciendo camino. Si modificamos los parametros, podriamos reducir los pasos y aumentar los casos de exito. 
# 
# Os invitamos a probar convinaciones que den un resulatado mejor que este!

# ## 5 Resultados
# 
# Por último, procedemos a realizar pruebas en el entorno de test (env_test) utilizando la Q-Table previamente generada. Por lo tanto, evaluaremos el desempeño del agente observando cómo comporta en el entorno después del entrenamiento en condiciones reales de prueba, utilizando la política óptima almacenada en la Q-Table.
# 
# Realizaremos las pruebas validando el comportamiento del agente en un total de 5 episodios  5 de prueba, imprimiendo detalles del progreso en cada uno, donde esperamos que la recompense incremente según se vayan consumiendo los intentos (episodios).
# 
# Nota: Al ejecutar la siguiente celda podremos ver de foma visual el rendimiento de nuestro algoritmo en una nueva ventana emergente.

# In[106]:


env_test.reset()

for episode in range(5):
    state, info = env_test.reset()
    step = 0
    total_reward = 0
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        action = np.argmax(q_table[state,:])
        
        new_state, reward, terminated, truncated, info = env_test.step(action)
        
        total_reward += reward
        if terminated or truncated:       
            print("Number of steps", step)
            print("Reward: ", total_reward)
            break

        state = new_state
env_test.close()


# Una vez ejecutado el código, se observa que el agente ha aprendido una política razonablemente buena, pero analicemos los detalles para determinar su calidad.
# 
# Por una parte, se han obtenido <strong><u>recompensas altas</u></strong> en cada episodio (0.96, 0.969, 0.979, 0.974), ya que se tratan de valores cercanos de 1, lo que sugiere que el agente está cumpliendo con su objetivo de maximizar la recompensa acumulada. Esto es un indicador positivo de que ha aprendido a navegar el entorno correctamente.
# 
# Por otra parte, el <strong><u>número de pasos es razonable</u></strong> para completar el entorno de pruebas, además de que presentan valores decrecientes según el avance de los episodios (40, 31, 21, 26). Aunque no hay un patrón completamente consistente, el agente parece necesitar menos pasos en episodios posteriores para alcanzar el objetivo, lo que sugiere una mejora en su eficiencia.
# 
# Por lo tanto, se observa que los resultados del modelo son buenos y reflejan un aprendizaje efectivo del agente.
# 
# 
# 
# #### ¡Muchas gracias por leer!
