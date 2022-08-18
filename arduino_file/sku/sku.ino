int b1=0,b2=0,b3=0;
int count=0;

    #if defined(ARDUINO) && ARDUINO >= 100
    #include "Arduino.h"
    #else
    #include "WProgram.h"
    #endif

    #include "EMGFilters.h"

    #define TIMING_DEBUG 1

    #define SensorInputPin A0 // input pin number

    EMGFilters myFilter;
    // discrete filters must works with fixed sample frequence
    // our emg filter only support "SAMPLE_FREQ_500HZ" or "SAMPLE_FREQ_1000HZ"
    // other sampleRate inputs will bypass all the EMG_FILTER
    int sampleRate = SAMPLE_FREQ_1000HZ;
    // For countries where power transmission is at 50 Hz
    // For countries where power transmission is at 60 Hz, need to change to
    // "NOTCH_FREQ_60HZ"
    // our emg filter only support 50Hz and 60Hz input
    // other inputs will bypass all the EMG_FILTER
    int humFreq = NOTCH_FREQ_50HZ;

    // Calibration:
    // put on the sensors, and release your muscles;
    // wait a few seconds, and select the max value as the threshold;
    // any value under threshold will be set to zero  校准
    static int Threshold = 0;

    int outdata;
  
    unsigned long timeStamp;
    unsigned long timeBudget;

    void setup() {
        /* add setup code here */
        myFilter.init(sampleRate, humFreq, true, true, true);

        pinMode(2,INPUT);
        
        // open serial
        Serial.begin(115200);

        // setup for time cost measure
        // using micros()
        timeBudget = 1e6 / sampleRate;
        // micros will overflow and auto return to zero every 70 minutes
    }

    void loop() {
        /* add main program code here */
        // In order to make sure the ADC sample frequence on arduino,
        // the time cost should be measured each loop
        /*------------start here-------------------*/
        timeStamp = micros();

        int Value = analogRead(SensorInputPin);
//        Serial.println(Value);
//        delay(10);
        
        // filter processing
        int DataAfterFilter = myFilter.update(Value);

        int envlope = sq(DataAfterFilter);
        // any value under threshold will be set to zero
        envlope = (envlope > Threshold) ? envlope : 0;

        b1 = digitalRead(2);
        if(b1 != b2){
          if(b1 == LOW){
          count++;
          if(count!=1){
          Serial.println();
          }
          }
        }
        b2 = b1;
        if(count % 2 != 0){
            if (TIMING_DEBUG) { 
//              Serial.print("Squared Data: ");
              Serial.println(envlope);
              delayMicroseconds(500);
            }
          }
        timeStamp = micros() - timeStamp;

    }
