# -*- encode: utf-8 -*-

import turtle as t
t.speed(10)
t.pensize(5)
#脸
t.showturtle()
t.pencolor("orange")
t.begin_fill()
t.penup()
t.goto(0,-200)
t.pendown()
t.fillcolor("Yellow1")
t.circle(200)
t.end_fill()
#嘴
t.pencolor("brown")
t.penup()
t.goto(0,-175)
t.pendown()
t.circle(170,80)
t.left(180)
t.circle(-170,160)
t.seth(0)
#眼睛
t.pensize(3)
 #右眼框
t.penup()
t.goto(30,75)
t.pendown()
t.seth(40)
t.fillcolor("white")
t.begin_fill()
t.circle(-110,95)
pos1=t.pos()
#print(pos1)
t.penup()
t.goto(30,75)
t.pendown()
t.seth(-140)
t.circle(18,180)
t.circle(-80,80)
t.goto(pos1)
t.end_fill()
 #右眼珠
t.pencolor("black")
t.penup()
t.goto(45,60)
t.pendown()
t.begin_fill()
t.fillcolor("black")
t.circle(8)
t.end_fill()
 #右眉毛
t.penup()
t.goto(60,105)
t.pendown()
t.seth(60)
t.circle(-40,110)


 #左眼框
t.pencolor("brown")
t.penup()
t.goto(-170,75)
t.pendown()
t.begin_fill()
t.fillcolor("white")
t.seth(40)
t.circle(-110,95)
pos2=t.pos()
t.penup()
t.goto(-170,75)
t.pendown()
t.seth(-140)
t.circle(18,180)
t.circle(-80,80)
t.goto(pos2)
t.end_fill()
 #左眼珠
t.pencolor("black")
t.penup()
t.goto(-155,60)
t.pendown()
t.begin_fill()
t.fillcolor("black")
t.circle(8)
t.end_fill()

#左眉毛
t.penup()
t.goto(-60,105)
t.pendown()
t.seth(120)
t.circle(40,110)


t.hideturtle()
t.done()




