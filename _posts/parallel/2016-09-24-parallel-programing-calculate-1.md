---
title: CUDA로 배워보는 병렬 프로그래밍(Parallel Programming) - 2
categories: [Parallel]
tags: [parallel, cuda]
description: 병렬 프로그래밍의 기본적인 연산 방법 및 원리
date : 2016-09-24 22:00:00 +0900
---

`CUDA`는 `C++ like Language`이다. 사실상, C++에서 `CUDA` 관련 라이브러리가 추가되어 있는 구조라고 생각하면 쉽다. 그래서 `CUDA` 파일인 `.cu` 소스파일을 실행시키면 다음 사진과 같이 실행을 시킨다.
[!logic](https://c4.staticflickr.com/9/8230/29267752363_60e1c95eca_b.jpg)
그리고 CPU가 실행시키는 코드는 `serial 코드(host)`라고 부르고, GPU가 실행시키는 코드는 `Kernel 코드(device)`라고 부른다. CPU가 실행히키는 코드는 순차적으로 실행시키고 `CUDA` 코드를 실행시키면 그때부터 병렬로 실행되고, CPU는 병렬 코드가 끝날 때까지 기다리는 구조이다.

## 데이터 전송
`CUDA` 및 다른 병렬 프로그래밍을 할 때는 `GPU`를 사용를 하여야 한다. 하지만, 이 `GPU`를 사용하기 위해선 `GPU` 안에 있는 `VRAM`으로 처리할 데이터를 옮겨나야 한다. 그렇기 때문에 병렬 프로그래밍을 공부할 때 가장 먼저 배워야하는 것이 `DRAM -> VRAM`, `VRAM -> DRAM`, `VRAM -> VRAM`과 같이 데이터를 옮기는 것이다.

데이터 전송을 하기 전에 `CUDA`에서는 몇 가지 규칙이 있다.

* 모든 `CUDA` library 함수들은 `cuda`로 시작하다.
* Library 함수들은 error code 혹은 `cudaSucess`를 반환한다.
* GPU에서 사용할 변수들은 앞에 `dev_`를 붙여야한다.

그러면 간단하게 배열을 `GPU`를 거쳐서 카피하는 코드를 작성해보면 다음과 같다.

{% highlight c++ %}
#include <cstdio>

int main(void) {
    //host 데이터
    const int size = 2;

    int a[size] = {2, 1};
    int b[size] = {0, 0};

    printf("{ %d, %d }\n", b[0], b[1]);

    // device 데이터
    int *dev_a = 0, *dev_b = 0;

    // VRAM에 메모리 할당 (c언어의 malloc과 동일 역활)
    cudaMalloc((void**)&dev_a, size*sizeof(int));
    cudaMalloc((void**)&dev_b, size*sizeof(int));

    // host -> device로 카피
    cudaMemcpy(dev_a, a, size*sizeof(int), cudaMemcpyHostToDevice);
    // device -> device 카피
    cudaMemcpy(dev_b, dev_a, size*sizeof(int), cudaMemcpyDeviceToDevice);
    // device -> host 카피
    cudaMemcpy(b, dev_b, size*sizeof(int), cudaMemcpyDeviceToHost);

    // 메모리 할당 해제
    cudaFree(dev_a)
    cudaFree(dev_b);

    printf("{ %d, %d }\n", b[0], b[1]);

    return 0;
}
{% endhighlight %}

`cudaMalloc`은 메모리 할당을 위해 쓰고, 첫번째 파라미터로는 할당할 포인터, 두 번째 파라미터로는 할당할 사이즈가 들어간다. 성공시에는 `cudaSuccess` 실패시에는 `cudaErrorMemoryAllocation`을 반환한다.

`cudaMemcpy`는 메모리를 복사를 할 때 사용을 하고, 첫 번째 파라미터는 목적지 주소의 포인터, 두 번째 파라미터는 복사할 주소의 포인터, 세 번째는 복사할 크기, 마지막은 `cudaMemcpyKind`인데 이것은 (`cudaMemcpyHostToHost`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToDevice`, `cudaMemcpyDeviceToHost`) 이다.
그리고 `cudaMemcpy`는 카피가 끝날 때까지 CPU 쓰레드를 블록시킨다.

`cudaFree`는 free와 동일하게 메모리를 해제시킨다.
