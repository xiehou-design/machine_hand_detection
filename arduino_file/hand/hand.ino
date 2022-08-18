#include <Servo.h>

/*
 * 9 食指
 * 10 中指
 * 11 小智
 * 角度小往里
 * 
 * 3.6 大拇指关节
 * 2.7 大拇指上关节
 */

Servo myservo1;
Servo myservo2;
Servo myservo3;


int flag = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  myservo1.attach(11);
  myservo2.attach(10);
  myservo3.attach(9);

  pinMode(3,OUTPUT);
  pinMode(6,OUTPUT);
  analogWrite(6,0);
  analogWrite(3,90);
  delay(500);
  analogWrite(6,0);
  analogWrite(3,0);
  
  pinMode(7,OUTPUT);
  pinMode(2,OUTPUT);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0){
  flag = Serial.read();
  Serial.println(flag);
  if(flag == '3'){
      gesture1();
      flag = '0';
    }
  else if(flag == '5'){
      gesture2();
      flag = '0';
    }
  else if(flag == '0'){
      gesture3();
      flag = '0';
    }
  else if(flag == '1'){
      gesture4();
      flag = '0';
    }
   else if(flag == '4'){
      gesture5();
      flag = '0';
    }
  else{
      stopp();
    }

  }
}

void stopp(){
    analogWrite(6,0);
    analogWrite(3,0);
    analogWrite(7,0);
    analogWrite(2,0);
  }

void gesture1(){
    myservo1.write(0);
    myservo2.write(90);
    myservo3.write(90);
    analogWrite(6,90);
    analogWrite(3,0);
    delay(750);
  }

void gesture2(){
    myservo1.write(90);
    myservo2.write(90);
    myservo3.write(90);
    analogWrite(6,0);
    analogWrite(3,90);
    delay(500);
  }

void gesture3(){
    myservo3.write(0);
    myservo2.write(90);
    myservo1.write(90);
    analogWrite(6,90);
    analogWrite(3,0);
    delay(500);
  }

void gesture4(){
    myservo3.write(90);
    myservo2.write(0);
    myservo1.write(0);
    analogWrite(6,0);
    analogWrite(3,90);
    delay(250);
  }

void gesture5(){
    myservo3.write(90);
    myservo2.write(90);
    myservo1.write(90);
    analogWrite(6,90);
    analogWrite(3,0);
    delay(750);
  }

void gesture6(){
    myservo1.write(0);
    myservo2.write(90);
    myservo3.write(90);
    analogWrite(6,90);
    analogWrite(3,0);
    delay(750);
  }
