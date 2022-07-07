from tensorflow.keras.layers import Conv2D, BatchNormalization, Add
import tensorflow as tf


# Entrada: Caracteristicas      (entrada)
#          Num. de canales      (num_kernel)
#          Radio de reducción   (ratio)
# Salida: Atención del módulo

def CBAM(entrada, num_kernel, ratio):
  reduccion= int(num_kernel//ratio)
  
  # Modulo de atención de canal
  favr= tf.reduce_mean(entrada, axis= [-1], keepdims= True)
  fmax= tf.reduce_max(entrada, axis= [-1], keepdims= True)
  
  davr_r= tf.keras.layers.Dense(reduccion, activation= 'relu')(favr)
  dmax_r= tf.keras.layers.Dense(reduccion, activation= 'relu')(fmax)
  
  davr_a= tf.keras.layers.Dense(num_kernel, activation= 'relu')(davr_r)
  dmax_a= tf.keras.layers.Dense(num_kernel, activation= 'relu')(dmax_r)
  
  x= tf.add(davr_a, dmax_a)
  x= tf.keras.layers.Activation('sigmoid')(x)
  
  #Multiplicación del resblock 1
  r= tf.math.multiply(entrada, x)

  #Modulo de atención espacial
  favr_s= tf.reduce_mean(r, axis= [-1], keepdims= True)
  fmax_s= tf.reduce_max(r, axis= [-1], keepdims= True)
  c= tf.concat([favr_s, fmax_s], axis= -1)
  c= Conv2D(1, 7, padding= 'same')(c)
  c= BatchNormalization()(c)
  c= tf.keras.layers.Activation('sigmoid')(c)

  #Multiplicacion del resblock 2
  salida= tf.math.multiply(x, c)

  return Add()([entrada, salida])