---
title: 동적계획법(Dynamic Programming) 시작하기 - 2
categories: [Algorithm]
tags: [dynamic programming]
description: 기본적인 1~2차원 다이나믹 프로그래밍에 대한 설명과 풀이
---

다이나믹 프로그래밍은 문제를 많이 풀어봐야지 느낌이 오는 것 같다. 그래서 이 글에서는 가장 기초적인 다이나믹 프로그램을 활용해야하는 문제에 내가 했던 접근을 설명할 생각이다.

## 계단오르기
### [2579번 계단오르기](https://www.acmicpc.net/problem/2579)

우선, 이 문제의 설명을 보면 다음과 같은 제약들이 있다.

* 계단은 한 번에 한 계단씩 또는 두 계단씩 오를 수 있다. 즉, 한 계단을 밟으면서 이어서 다음 계단이나, 다음 다음 계단으로 오를 수 있다.
* 연속된 세 개의 계단을 모두 밟아서는 안된다. 단, 시작점은 계단에 포함되지 않는다.
* 마지막 도착 계단은 반드시 밟아야 한다.

이 조건을 모두 만족시키는 경우 중에서도 최대의 값으로 가는 경우의 점수를 구해야한다.

다이나믹 프로그래밍에선 중요한 것은 다음과 같다.

* 이미 계산했던 값은 계산하지 말자.
* 문제의 제약을 이해한다.

이 문제에서 어떨 때, 중복 계산하는 경우가 발생하는지 생각을 해보자.

생각을 해보면 다음과 같은 경우 항상 같은 값이 계산되는 것을 알 수 있다.
> 지금 N번째 계단에 있는데, 지금까지 K개의 연속된 계단을 밟은 상태이다.

중복이 발생하는 경우를 찾았으니, 이제 이 경우를 점화식으로 나타내어야한다. 점화식은 `D[N][1] = max(D[N-2][1], D[N-2][2]) + (N번째 계단의 점수)`와 `D[N][2] = D[N-1][1] + (N번째 계단의 점수)`이다.

이 식을 반복문을 이용한 동적계획법으로 풀면 다음과 같다.

{% highlight c %}
#include <cstdio>

int n, stair[301], dp[301][3];
int max(int a, int b) { return (a>b)?a:b; }

int main()
{
    scanf("%d", &n);
    for(int i = 1; i<=n; i++)
        scanf("%d", &stair[i]);

    dp[0][0] = dp[0][1] = 0;
    dp[1][1] = dp[1][2] = stair[1];

    for(int i =2; i<=n; i++)
    {
        dp[i][2] = dp[i-1][1] + stair[i];
        dp[i][1] = max(dp[i-2][1], dp[i-2][2]) + stair[i];
    }
    printf("%d\n", max(dp[n][1], dp[n][2]));
}
{% endhighlight %}

## 1로 만들기
### [1463번 1로 만들기](https://www.acmicpc.net/problem/1463)

어떤 숫자 `X`가 주어져 있을 때, 1로 만드는 최소의 연산 횟수를 구하는 문제이다.
이 문제 같은 경우의 조건은 다음과 같다.

* X가 3으로 나누어 떨어지면, 3으로 나눈다.
* X가 2로 나누어 떨어지면, 2로 나눈다.
* 1을 뺀다.

이 문제도 한 번 곰곰히 어떻게 접근해야할지 생각해본다.



이 경우 해당 숫자에 대해서 계속 최소값으로 갱신시키면 된다. 다음과 같이 점화식을 세울수 있다.
`X가 3으로 나누어지면 D[X] = min(D[X], D[X-3] + 1)`, `X가 2로 나누어지면 D[X] = min(D[X], D[X-2] + 1)`, `모든 경우 D[X] = min(D[X], D[X] + 1)`이다.

{% highlight c %}
#include <cstdio>
#include <algorithm>
using namespace std;

int D[10000000];
int main() {
	int n;
	scanf("%d", &n);

	D[0] = D[1] = 0; // 초기값

	for (int i = 2; i <= n; i++) {
		int ret = 2e9;
		if (i % 3 == 0) ret = min(D[i / 3], ret);
		if (i % 2 == 0) ret = min(D[i / 2], ret);
		D[i] = min(ret, D[i - 1]) + 1;
	}
    printf("%d\n", D[n]);

}
{% endhighlight %}
