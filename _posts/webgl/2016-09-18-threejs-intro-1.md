---
title: Three.js로 시작하는 WebGL - 1
categories: [WebGL]
tags: [threejs, webgl]
description: 웹에서 그래픽을 보여주는데 쓰이는 WebGL을 threejs를 이용해서 Graphic을 웹에 그려보다. teamtreehouse의 튜토리얼을 따라해보자.
---
나는 프로젝트에서 3D 모델 stl파일을 웹에서 보여주는 작업을 해야하기 때문에 threejs를 배우기 시작하였다. 이제 threejs 관련 몇몇 글을 번역하면서
배운다는 생각으로 글도 포스팅 해 볼생각이다.

[ㅅ샘플코드](http://treehouse-code-samples.s3.amazonaws.com/threejs_logo_example/threejs_logo_example.zip);

## 설치 및 HTML파일 생성
우선 `Three.js`를 사용하기 위해서 다운로드를 받아야한다. (http://threejs.org/) 에서 다운을 받고, 다운 받을 폴더를 작업하고자 하는 폴더안에 삽입한다. 다운 받은 폴더 안에 build폴더와 example 폴더가 있을 것이다. 편의를 위해서 나같은 경우 각 폴더의 파일들을 js폴더를 만들어서 넣었다. 그래서 여기에 작성된 코드들은 js폴더가 있다는 가정하에 작성하겠다.

 그러면 프로젝트 폴더안에 `index.html` 파일을 다음과 같이 만든다.

{% highlight html %}
<!doctype html>
<html>
<script src ="js/three.min.js"></script>
<script src ="js/orbitControls.js"></script>
<script>
    var scene, camera, renderer;
    init();
    animate();
    <!-- 여기에는 3D 코드 작성 -->
</script>
</html>
{% endhighlight %}

## Scene 만들기
`Three.js`에서는 `scene` 개념을 정의했다. `scene`은 우리가 넣고자 객체(`Camera`, `Geometry`, `Lights`...) 하는 곳의 위치를 정의하는 것이다.
그리고 이미지를 보여줄 화면의 크기를 구하기 위해서 brower의 window의 크기는 `WIDTH`와 `HEIGHT`에 저장한다.

{% highlight javascript %}
function init() {

    scene = new THREE.Scene();
    var WIDTH = window.innerWidth;
    var HEIGHT = wndwo.innerHeight;
}
{% endhighlight %}


## Render 만들기
Render는 그래픽을 실제로 화면에 보여주는 일을 한다.
canvas 및 SVG의 Render를 사용할 수 있지만, GPU를 활용하여 빠르게 Render하기 위해서 `WebGL renderer`를 활용한다.

{% highlight javascript %}
function init() {
    //이전 코드

    renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(WIDTH, HEIGHT);  // Render할 크기
    document.body.appendChild(renderer.domElement); //HTML에 삽입

    //다음 코드
}
{% endhighlight %}
Javascript로 DOM을 html body안에 삽입한다. `renderer.domElement` 호출로 Three.js는 `canvas`를  body안에 삽입할 것 이다.

## Camera 만들기
이제 `Camera`를 만들어야할 차례이다. 우리는 `PerspectiveCamera`를 쓸 것이고 이는 다음과 같은 파라미터들을 가진다.

* FOV - 보는 각도
* Apsect - aspect ratio를 구하기 위해서 brower width 에서 broser height를 나눈다.
* Near - scene object와 Camera의 거리를 나타낸다.
* Far - 볼 수 있는 거리를 지정한다. 이 거리를 넘는 객체들은 보이지 않느다.

`Camera`를 생성한 다음에는 카메라의 위치를 (X, Y, Z) 좌표로 지정해야한다. default는 (0, 0, 0)이고, 이 예제에서는 Mesh와 Camera사이에 거리를 두기 위해서 (0, 6, 0)으로 지정한다. 그리고 마지막으로 Camera를 Scene에 추가하면 된다.

{% highlight javascript %}
function init() {
    //이전 코드

    camera = new THREE.PerspectiveCamera(45, WIDTH / HEIGHT, 0.1, 20000);
    camera.position.set(0, 6, 0);
    scene.add(camera);

    //다음 코드
}
{% endhighlight %}

## Lighting 추가

`PointLight` 객체를 `Scene`에 추가를 해야하만 우리가 추가한 3D Object를 볼 수 있도록 된다.(Light는 사방을 비춘다.) `PointLight`는 다음 과 같은 Parameter를 같는다.

* `color` - Light의 RGB값의 16진수 표현
* `intensity` - Light의 세기
* `distance` - Light의 길이
* `decay` - Light가 가면서 어두어지는 정도

{% highlight javascript %}
function init() {

    //이전 코드

    // scene의 배경 색
    renderer.setClearColor(0x333F47, 1);

    var light = new THREE.PointLight(0xffffff);
    light.position.set(-100,200,100);
    scene.add(camera);

    //다음 코드
}
{% endhighlight %}

## Controls 추가

이 Controls는 처음에 `index.html`에 추가한 `orbit controls` 안의 함수이다.
반드시 해야하는 것은 아니지만, 이는 우리가 마우스 조작으로 `Mesh`와 `Orbit`을 조작할 수 있게 해준다.

{% highlight javascript %}
function init() {

    //이전 코드
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    //다음 코드

}
{% endhighlight %}

## Scene Render하기

초기화 함수를 `init()`을 실행하고 난 다음, 애니메이션 함수인 'animation()'을 실행 시켜야한다. 왜냐하면, 위의 `Controls`에서 추가한 조작을 할 때마다 새로 `Mesh`를 다시 `Render`를 해주어야 하기 때문이다.

`requestAnimationFrame()` 함수는 여러 장점이 있지만, 불필요하게 다시 애니메이션을 그리는 일을 방지해주는 기능을 한다. 이 함수를 재귀 구조로 돌리면서 다시 `Mesh`를 그려준다.

{% highlight javascript %}
function init() {

    //이전 코드

    requestAnimationFrame(animate);

    // Render The Scene
    renderer.render(scene, camera);
    controls.update();
}
{% endhighlight %}

### 출처
[TreeHouse](http://blog.teamtreehouse.com/the-beginners-guide-to-three-js)
