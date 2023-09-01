import os
import glob
import argparse
import matplotlib
import numpy as np

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

# Loop para salvar cada "camada" da matriz em um arquivo separado
#output_folder = '/content/DD2/Medidas/'
#for i, output_layer in enumerate(outputs):
#    output_filename7 = os.path.join(output_folder, f'output_layer_{i}.txt')
#    np.savetxt(output_filename7, output_layer.flatten(), fmt='%f')
#    print(f"Camada de saída {i} salva no arquivo '{output_filename7}'")
#print('...Ending loop')
# Loop para salvar cada "camada" da matriz em um arquivo separado
# Loop para salvar cada "camada" da matriz em um arquivo separado
output_folder = '/content/DD2/Medidas/'
for i, output_layer in enumerate(outputs.transpose(3, 0, 1, 2)):  # Transpose para iterar ao longo da dimensão correta
    output_filename = os.path.join(output_folder, f'output_layer_{i}.txt')
    np.savetxt(output_filename, output_layer.flatten(), fmt='%f')
    print(f"Camada de saída {i} salva no arquivo '{output_filename}'")
print('...Ending loop')




# Salvar a matriz em um arquivo de texto
#output_filename = 'output_matrix.txt'
#np.savetxt(output_filename, outputs.flatten(), fmt='%f')
print("... .begin. ...")

#output_filename = '/content/DD2/output_matrixVF.txt'
output_filename7 = '/content/DD2/'
output_filename3 = '/content/DD2/output_matrixVF3.txt'
np.savetxt(output_filename3, outputs.flatten(), fmt='%f')
#np.savetxt(output_filename, str(outputs), fmt='%f')



print('Loading loop...')

print(f"Matriz de saída salva no arquivo '{output_filename7}'")

print("... .end. ...")
#f=open("/content/DD2/PredictionMatrix.txt", "w")
#f.write(str(outputs))
#f.close()

#print(f"Imprimindo matriz: {str(outputs)} /fim)
print("Imprimindo...")
print(str(outputs))
print(" ...Fim")
#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
