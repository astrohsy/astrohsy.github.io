---
title: 쉬운 다이나믹프로그래밍 문제 및 풀이
categories: [algorithm]
tags: [dp, algorithm]
description: 입문용으로 적합한 다이나믹 프로그래밍(동적계획법) 문제들을 모아두었습니다.
---

혼자 이것 저것 문제들을 풀어보면서 처음 시작할 때, 공부하기 쉬운 DP문제들을 올릴려고 합니다.
이 글은 계속 업뎃할 계획이다. 5개 문제가 넘어가면 새로운 포스트로 다시 작성할 생각이다.

## [오르막 길](https://www.acmicpc.net/problem/2846)
LIS(Longest Increasing sub-sequence)라는 DP알고리즘이 있는데, 이 문제는 LIS의 축소판인 것 같다.
간단하게 O(n) time안에 풀린다. 즉 for문 하나만으로 풀린다는 소리이다.

----

{% highlight c++ %}
#include <cstdio>
int D[1001];
int main() {
    int n;
    scanf("%d", &n);
    int _max = 0;
    int num, prev = 2e9;
    for(int i= 1; i<=n; i++) {
        scanf("%d", &num);
        if(prev < num) D[i] = D[i-1] + num - prev;
        if(_max < D[i]) _max = D[i];
        prev = num;
    }

    printf("%d\n", _max);

    return 0;
}
{% endhighlight %}
