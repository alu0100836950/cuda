# Práctica de cuda

##### Alberto Martín Núñez


## Objetivo de la práctica

Desarrolla una versión en CUDA del código que se ha desarrollado en la práctica 3 para el procesamiento de imágenes.

## Version de CUDA

Para realizar esta version vamos a detallar los pasos principales que he hecho:

- Primero recogemos la imagen que queremos procesar. Esta imagen se pasa por línea de comandos.
- Posteriormente creamos una matriz resultante que tendrá las propiedades *rows* y *cols* de la imagen original.

```c++
   Mat *imagen = new Mat(imread(argv[1], IMREAD_GRAYSCALE));
    
    
    cout << "La imagen mide " << imagen->rows << " x " << imagen->cols << " pixeles" << endl;
    Mat *resultCPU = new Mat(imagen->rows, imagen->cols, CV_8UC1);
```

- Creamos los punteros auxiliares:

```c++
    
    uchar * src;
    uchar * result;
    uchar * kernel_;
```

- Reservamos memoria en las variables auxiliares con los tamaños de cada matriz y seleccionamos el numero debloques a utilizar y los hilos por bloque.

```c++
cudaMalloc(&src, imagen->total() * sizeof(uchar));
    cudaMalloc(&result, imagen->total() * sizeof(uchar));
    cudaMalloc(&kernel_, kernel->total() * sizeof(uchar));

    
    int num_B = 16;
    dim3 threadsPerBlock(imagen->rows / num_B);
```
- Ahora podemos ejecutar nuestro filtro y para ello tenemos que hacer una serie de llamadas:

```c++

    //Enviamos al dispositivo la iamgen y el kernel
    cudaMemcpy(src, imagen->data, imagen->total() * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_, kernel->data, kernel->total() * sizeof(uchar), cudaMemcpyHostToDevice);
    
    //aplicamos el filtro usando los parametros que vamos a usar en la funcion eindicandole los bloques y el numero de hilos previamente asignados
    applyFilter<<<num_B, threadsPerBlock>>>(src, result, imagen->rows, imagen->cols, kernel_);

    //devolvemos los datos obtneidos dee aplicar el filtro a la imagen
    cudaMemcpy(resultCPU->data, result, imagen->total(),cudaMemcpyDeviceToHost);

```

- Una vez aplicado el filtro la imagen se queda guarda segun el segundo parámetro pasado por línea de comandos.


## Ejecucion del programa

Para ejecutar esta version de cuda para el procesamiento de imagenes tenemos que relizar lo siguiente:

1. `Make cuda`
2. `./cuda.run image.jpg output.jpg` o `Make run`



## Ejemplo de ejecución

Aqui mostramos la imagen original y la imagen de salida, asi como el tiempo que se ha tardado en hacer el procesamiento:

```
La imagen mide 630 x 630 pixeles
Usando dispositivo CUDA:  GeForce GTX 1650 SUPER
Ejecutamos el filtro
Tiempo de ejecucion: 0.001 sec
```

![lena original](lena.jpg)

![lena salida](file__output.jpg)


## Comparacion entre los anteriores algoritmos y cuda

### Imagen pequeña

| Version    | Tiempo(segundos) | Tamaño de la imagen |
|------------|--------|---------------------|
| Secuencial | 0.13   | 630 x 630           |
| CUDA       | 0.001  | 630 x 630           |

### Imagen más grande

| Version    | Tiempo(segundos) | Tamaño de la imagen |
|------------|------------------|---------------------|
| Secuencial | 3.14             | 3000 x 4000         |
| MPI        | 0.5 (8 cores)    | 3000 x 4000         |
| OpenMP     | 0.45             | 3000 x 4000         |
| CUDA       | 0.026            | 3000 x 4000         |


## Conclusiones

Resulta muy interesante como la gráfica tiene una capacidad bastante mayor para ejecutar el procesamiento de imágenes. Es cierto que la programación y el entendimiento de esta versión es más complejo pero cuando obtienes los resultados resulta bastante eficiente indagar y aprender sobre esta metodología de paralelizar programas usando la tarjeta gráfica con **CUDA**.

