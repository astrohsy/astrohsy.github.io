---
title: 깃허브(Github) 기초적인 사용법
categories: [Github]
tags: [github, project]
description: 깃허브를 처음으로 사용하는 사람에게 기초적인 사용방법에 대한 설명 제공
---

최근 학교에서 강제로 Github를 쓰도록하고 있다... 그래서 이 포스트에선 깃허브를 배우면서 알게된 기초적인 사용설명을 설명해보고자 한다.
우선 왜 깃허브를 쓰는지에 대해 말하자면 프로젝트를 진행할 때, 반드시 프로젝트의 버전 관리를 하여야한다. 그래야만 나중에 프로젝트 진행 중에 문제가 생길시 되돌릴 수가 있고, 프로젝트가 어느정도로 진행되었는지 알 수 있기 때문이다.  

## 깃(Git)
이런 버전 관리 툴은 이미 많이 존재하지만, 최근에 가장 많이 사용하는 툴은 깃(Git)이다. 그 이유는 다른 버전 관리 툴과 다르게 "분산(Distributed)"을 하기 때문이다. 분산을 통해 버전 기록과 통합을 별도로 진행을 해서 프로젝트의 자유도를 높일 수가 있다고 한다. 각자의 버전 기록을 할 때는 commit을 하고, 통합이 필요할 때만 push를 함으로써 기록하면 되기 때문이다.

## 깃허브(Github)
깃허브는 깃을 이용한 프로젝트를 웹에 호스팅할 수 있도록 하는 것이다. 깃 자체는 Local에 기록을 하는데, 깃허브를 사용함으로써 이제 한 자리에 없어도 같이 프로젝트를 진행할 수 있도록 된 것이다. 다른 호스팅 사이트도 있지만, 깃허브가 가장 유명하다.

## 기초적인 사용 방법
만약에 git command가 없다고 뜬다면 [이곳에서 설치한다](https://git-scm.com/)
#### 1. 새로운 프로젝트를 만들 때
* 프로젝트를 진행하기 전에 깃허브에 들어가서 repository를 만들어야한다.
깃허브에서 New repository 버튼을 눌러 repository를 만든다.
* 생성된 repository안에 들어가서 초록색 clone or download 버튼을 클릭한 다음 나타는 링크를 복사한다.
* 프로젝트를 만들고 싶은 경로를 Command 창으로 들어간다.
* 복사한 주소가 만약 "https://github.com/astrohsy/blog.git"이라면 다음을 커맨드창에 입력한다.{% highlight sh %} git clone https://github.com/astrohsy/blog.git {% endhighlight %}

### 2. 기존 프로젝트를 깃허브에 올릴 때
* 프로젝트를 진행하기 전에 깃허브에 들어가서 repository를 만들어야한다.
깃허브에서 New repository 버튼을 눌러 repository를 만든다.
* 생성된 repository안에 들어가서 초록색 clone or download 버튼을 클릭한 다음 나타는 링크를 복사한다.
* 프로젝트를 만들고 싶은 경로를 Command 창으로 들어간다.
* 복사한 주소가 만약 "https://github.com/astrohsy/blog.git"이라면 다음을 커맨드 창에 입력한다.
{% highlight sh %}
git remote add origin https://github.com/astrohsy/blog.git
git add .
git commit -am "first commit"
git push origin master
{% endhighlight %}

### 3. 자신의 변경이력을 저장할 때
* 프로젝트 폴더안에 들어가서 Command 창에 들어간다.
* 만약에 새로 생성한 파일이 있을 경우
{% highlight sh %}
git add . # 만약에 특정 파일만 추가하고 싶으면 해당 파일의 경로를 적는다.
git commit -am "여기에 수정 내역을 적는다"
{% endhighlight %}
* 따로 없는 경우에는
{% highlight sh %}
git commit -am "여기에 수정 내역을 적는다"
{% endhighlight %}


### 4. 자신의 변경이력을 프로젝트랑 합칠 때
* 프로젝트 폴더안에 들어가서 Command 창에서
{% highlight sh %}
git push origin master
{% endhighlight %}

## 주의사항
* 같은 파일을 동시에 작업하지 않는 것이 좋다. 같이 하게되면 나중에 conflict가 생기는데 이것을 merge할려면 골치 아프다


### 참조자료
[버전 관리 시스템 유랑기, 그리고 Git 적응기](https://gist.github.com/benelog/2922437)

[Backlogtool git-guide](https://backlogtool.com/git-guide/kr/) <- 강추!
