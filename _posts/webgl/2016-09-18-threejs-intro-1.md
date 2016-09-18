---
title: threejs로 시작하는 WebGL - 1
categories: [WebGL]
tags: [threejs, webgl]
description: 웹에서 그래픽을 보여주는데 쓰이는 WebGL을 threejs를 이용해서 사용해보자.
---
나는 프로젝트에서 3D 모델 stl파일을 웹에서 보여주는 작업을 해야하기 때문에 threejs를 배우기 시작하였다. 이제 threejs 관련 몇몇 글을 번역하면서
배운다는 생각으로 글도 포스팅 해 볼생각이다.

## 설치 및 HTML파일 생성
우선 three.js를 사용하기 위해서 다운로드를 받아야한다. (http://threejs.org/) 에서 다운을 받고, 다운 받을 폴더를 작업하고자 하는 폴더안에 삽입한다. 다운 받은 폴더 안에 build폴더와 example 폴더가 있을 것이다. 편의를 위해서 나같은 경우 각 폴더의 파일들을 js폴더를 만들어서 넣었다. 그래서 여기에 작성된 코드들은 js폴더가 있다는 가정하에 작성하겠다.

 그러면 프로젝트 폴더안에 index.html 파일을 다음과 같이 만든다.

{% highlight html %}
<!doctype html>
<html>
<script src ="js/three.min.js"></script>
<script src ="js/orbitControls.js"></script>
<script>
    <!-- 여기에는 3D 코드 작성 -->
</script>
</html>
{% endhighlight %}

## Scene 만들기
Three.js에서는 ___scene___ 개념을 정의했다. ___scene___은 우리가 넣고자 객체(Camera, Geometry, Lights...) 하는 곳의 위치를 정의하는 것이다.
그리고 brower의 window의 크기는 __WIDTH__와 __HEIGHT__에 저장한다.

{% highlight javascript %}
function init() {

    scene = new THREE.Scene();
    var WIDTH = window.innerWidth;
    var HIEGHT = wndwo.innerHeight;
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

--- 작성중 ---

### 출처
[TreeHouse](http://blog.teamtreehouse.com/the-beginners-guide-to-three-js)
