---
title: CUDA로 배우는 병렬 프로그래밍(Parallel Programming) - 행렬(Matrix) 계산
categories: [Parallel]
tags: [parallel, cuda]
description: 병렬 프로그래밍의 기본적인 연산 방법 및 원리 및 행렬(Matrix)에 직접 적용, 행렬 곱셈 및 행렬 덧셈
date : 2016-10-01 22:00:00 +0900
---

우선 행렬 계산을 하기 전에 CUDA에서 어떤 방식으로 여러 개의 쓰레드를 돌리는지를 알아야한다. CUDA에선 `BLOCK`과 `GRID`로 쓰레드 그룹을 관리한다.

`BLOCK`에서는 여러 개의 `THREAD`를 (x, y, z) 즉, 3차원 이하로 가지고 있을 수 있고, `GRID`는 여러 `BLOCK`을 2차원 이하(x, y) 로 가지고 있을 수 있다. 그림으로 나타내면 다음과 같다. 그리고 한 `BLOCK`에는 1024개가 넘는 `THREAD`를 가지고 있을 수 없다.

![image](https://farm6.staticflickr.com/5469/30198898966_ee2eb65f07_c.jpg)

그리고 이와 같은 `BLOCK`과 `GRID`를 정의한 다음에 `Kernel Function` 즉 Device에서 돌아갈 함수랑 같이 쓰여야한다.

{% highlight c++ %}
__global__ void kernelFunc();

dim3 DimGrid(10, 10); //100 thread blocks
dim3 DimBlock(4, 5, 6); // 120 thread per Blocks

kernelFunc<<< DimGrid, DimBlock>>>();
{% endhighlight %}

위와 같이 C++에서 템플릿을 넘기듯이 사용하면 된다.
`dim3` 자료형 같은 경우 3개 미만으로 parameter를 넘기는 것도 가능하다.

위와 같이 정의하면 각 `thread`는 고유한 `blockIdx`와 `threadIdx`를 가진다.

위 두 개의 고유한 값을 이용해서 각 `thread`는 자기의 고유한 위치를 알 수 있다. 그리고 이렇게 정의된 `Blocks of Threads`는 CUDA 자체 Queue를 통해서 `SMs(Streaming multiprocessors)`에 할당되어 CUDA가 계산을 해준다.

## 행렬 덧셈 구현

{% highlight c++ %}
#include <cstdio>

__global__ void addKernel(int *c, const int *a, const int *b) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = y * (blockDim.x) + x; // 1차원 인덱스로 변환

    c[i] = a[i] + b[i];
}

int main() {
    //host data
    const int HEIGHT = 5;
    const int WIDTH = 5;

    int a[HEIGHT][WIDTH];
    int b[HEIGHT][WIDTH];
    int c[HEIGHT][WIDTH];

    // 배열 초기화
    for(int y = 0; y<HEIGHT; y++) {
        for(int x = 0; x<WIDTH; x++) {
            a[y][x] = y*10 + x;
            b[y][x] = (y * 10 + x) * 100;
        }
    }

    //device data
    int *dev_a = 0, *dev_b = 0, *dev_c = 0;

    //메모리 할당
    cudaMalloc( (void**)&dev_a, HEIGHT*WIDTH*sizeof(int));
    cudaMalloc( (void**)&dev_b, HEIGHT*WIDTH*sizeof(int));
    cudaMalloc( (void**)&dev_c, HEIGHT*WIDTH*sizeof(int));

    // host에서 device로 데이터 전송
    cudaMemcpy(dev_a, a, HEIGHT*WIDTH*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, HEIGHT*WIDTH*sizeof(int), cudaMemcpyHostToDevice);

    // GPU에서 행렬의 하나의 원소에 쓰레드 한 개씩 생성해서 연산
    dim3 dimBlock(HEIGHT, WIDTH, 1); // (x, y, z)
    addKernel<<<1, dimBlock>>>(dev_c, dev_a, dev_b);

    // device에서 host로 데이터 전송
    cudaMemcpy(c, dev_c, HEIGHT*WIDTH*sizeof(int), cudaMemcpyDeviceToHost);

    // 결과 출력
    for(int y = 0; y<HEIGHT; y++) {
        for(int x = 0; x<WIDTH; x++) {
            printf("%5d", c[y][x]);
        }
        printf("\n");
    }

    return 0;
}
{% endhighlight %}

## 행렬 곱셈 연산

행렬 곱셈 같은 경우는 O(n^3) 이다. 다음과 같이 구현해서 한 쓰레드 당 O(n)의 시간복잡도를 가지는 프로그램을 작성할 수 있다.

{% highlight c++ %}
__global__ void mulKernel(int *c, const int *a, const int *b, int N) {
    // global index로 변환
    int ROW = blockIdx.y*blockDim.y + threadIdx.y;
    int COL = blockIdx.x*blockDim.x + threadIdx.x;

    double sum = 0.0;

    /*
        이건 thread는 메모레에 Random Access를 하기 때문에
        thread가 불필요한 계산까지도 할 수 있다. 그래서 그것을
        방직하기 위해서 매트릭스에 속하지 않은 것들은 계산하지 않기
        위해서 조건문을 걸어둔 것 이다.
    */
    if( ROW < N && COL < N) {
        for(int k = 0; k<N; k++) {
            sum += a[ROW * N + k] * b[k * N + COL];
        }
    }
    c[ROW * N + COL] = sum;
}
{% endhighlight %}
## 출처
Programming Massively Parallel Processors(ECE498AL) by Davia Kirk and Win-mei W. Hwu

[QuantStart: MATRIX-MATRIX MULTIPLICATION ON THE GPU WITH NVIDIA CUDA](https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA)
