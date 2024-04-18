/*
 * Copyright (c) 2021-2022, ShanliTech
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2021-12-29     Hongtao Yao  first implementation
 *
 * Notes:
 * This is a keyword spotting example using NNoM
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define ALSA_PCM_NEW_HW_PARAMS_API
#include <alsa/asoundlib.h>

#include "nnom.h"
#include "kws_weights.h"

#include "mfcc.h"
#include "math.h"

#define SaturaLH(N, L, H) (((N)<(L))?(L):(((N)>(H))?(H):(N)))
#define _MAX(x, y) (((x) > (y)) ? (x) : (y))
#define _MIN(x, y) (((x) < (y)) ? (x) : (y))

#define SAMPLE_RATE 16000
#define SAMPLE_CHANNEL 1
#define AUDIO_FRAME_LEN (512) //31.25ms * 16000hz = 512, // FFT (windows size must be 2 power n)

// NNoM model
nnom_model_t *model = NULL;


mfcc_t * mfcc = NULL;

//the mfcc feature for kws
#define MFCC_LEN            (62)
#define MFCC_COEFFS_FIRST   (1)     // ignore the mfcc feature before this number
#define MFCC_COEFFS_LEN     (13)    // the total coefficient to calculate
#define MFCC_TOTAL_NUM_BANK (26)    // total number of filter bands
#define MFCC_COEFFS         (MFCC_COEFFS_LEN-MFCC_COEFFS_FIRST)
#define MFCC_FEAT_SIZE      (MFCC_LEN * MFCC_COEFFS)

// full 34 labels
const char label_name[][10] =  {
    "backward", "bed", "bird", "cat", "dog", "down", "eight","five", "follow", "forward",
    "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right",
    "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "yes", "zero", "unknow"
};

float   mfcc_features_f[MFCC_COEFFS];             // output of mfcc
int8_t  mfcc_features[MFCC_LEN][MFCC_COEFFS];     // ring buffer
int8_t  mfcc_features_seq[MFCC_LEN][MFCC_COEFFS]; // sequencial buffer for neural network input.
uint8_t mfcc_feat_index = 0;

// debugging controls
bool is_print_abs_mean = false; // to print the mean of absolute value of the mfcc_features_seq[][]
bool is_print_mfcc  = false;    // to print the raw mfcc features at each update

static int32_t abs_mean(int8_t *p, size_t size) {
    int64_t sum = 0;
    for(size_t i = 0; i<size; i++) {
        if(p[i] < 0)
            sum+=-p[i];
        else
            sum += p[i];
    }
    return sum/size;
}

static void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit) {
    float limit = (1 << int_bit);
    float d;
    for(uint32_t i=0; i<size; i++) {
        d = round(_MAX(_MIN(din[i], limit), -limit) / limit * 128);
        d = d/128.0f;
        dout[i] = round(d *127);
    }
}

static void mfcc_process(uint16_t* buffer, size_t size) {
    // MFCC
    // do the first mfcc with half old data(256) and half new data(256)
    // then do the second mfcc with all new data(512).
    // take mfcc buffer
    for(int i=0; i<2; i++) {
        mfcc_compute(mfcc, &buffer[i*AUDIO_FRAME_LEN/2], mfcc_features_f);

        // quantise them using the same scale as training data (in keras), by 2^n.
        quantize_data(mfcc_features_f, mfcc_features[mfcc_feat_index], MFCC_COEFFS, 3);

        // debug only, to print mfcc data on console
        if(is_print_mfcc) {
            for(int i=0; i<MFCC_COEFFS; i++)
                printf("%d ",  mfcc_features[mfcc_feat_index][i]);
            printf("\n");
        }

        mfcc_feat_index++;
        if(mfcc_feat_index >= MFCC_LEN)
            mfcc_feat_index = 0;
    }
}

int main(void) {
    uint32_t    label;
    float       prob = 0.0;
    uint32_t    last_mfcc_index = 0;
    int16_t     audio_buffer[AUDIO_FRAME_LEN+AUDIO_FRAME_LEN/2];

    memset(audio_buffer, 0, sizeof(int16_t)*(AUDIO_FRAME_LEN+AUDIO_FRAME_LEN/2));

    // create and compile the model
    model = nnom_model_create();

    // calculate 13 coefficient, use number #2~13 coefficient. discard #1
    // features, offset, bands, 512fft, 0 preempha, attached_energy_to_band0
    mfcc = mfcc_create(MFCC_COEFFS_LEN, MFCC_COEFFS_FIRST, MFCC_TOTAL_NUM_BANK, AUDIO_FRAME_LEN, 0.97f, true);
    if( mfcc == NULL ) {
        printf("mfcc_create failed\n");
        exit(-1);
    }

    snd_pcm_t *dev = NULL;
    snd_pcm_uframes_t buffer_frames;
    snd_pcm_uframes_t period_frames;

    /* Open PCM device for playback. */
    int ret = snd_pcm_open(&dev, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (ret < 0) {
        printf("%s snd_pcm_open failed! %s\n", __FUNCTION__, snd_strerror(ret));
        exit(-1);
    }

    ret = snd_pcm_set_params(dev, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, SAMPLE_CHANNEL, SAMPLE_RATE, 1, 64000);
    if (ret < 0) {
        printf("%s unable to set hw parameters: %s\n", __FUNCTION__, snd_strerror(ret));
        exit(-1);
    }

    ret = snd_pcm_get_params(dev, &buffer_frames, &period_frames);
    if (ret < 0) {
        printf("%s unable to get hw parameters: %s\n", __FUNCTION__, snd_strerror(ret));
        exit(-1);
    }

    printf("snd buffer frames:%d period frames:%d\n", buffer_frames, period_frames);

    while(1) {
        // memory move
        // audio buffer = | 256 byte old data |   256 byte new data 1 | 256 byte new data 2 |
        //                         ^------------------------------------------<|
        // copy old 256 frames data
        memcpy(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN], (AUDIO_FRAME_LEN/2)*sizeof(int16_t));

        // feed new 512 frames data
        ret = snd_pcm_readi(dev, &audio_buffer[AUDIO_FRAME_LEN/2], AUDIO_FRAME_LEN);

        // compute mfcc features
        mfcc_process(audio_buffer, AUDIO_FRAME_LEN);

        // wait for new mfcc feature data updated, then copy
        if( last_mfcc_index != mfcc_feat_index ) {
            // copy mfcc feature ring buffer to sequance buffer.
            last_mfcc_index = mfcc_feat_index;
            uint32_t len_first = MFCC_FEAT_SIZE - mfcc_feat_index * MFCC_COEFFS;
            uint32_t len_second = mfcc_feat_index * MFCC_COEFFS;
            memcpy(&mfcc_features_seq[0][0], &mfcc_features[0][0] + len_second,  len_first);
            memcpy(&mfcc_features_seq[0][0] + len_first, &mfcc_features[0][0], len_second);

            // debug only, to print the abs mean of mfcc output. use to adjust the dec bit (shifting)
            // of the mfcc computing.
            if(is_print_abs_mean) {
                printf("abs mean:%d\n", abs_mean((int8_t*)mfcc_features_seq, MFCC_FEAT_SIZE));
            }

            // ML
            memcpy(nnom_input_data, mfcc_features_seq, MFCC_FEAT_SIZE);
            nnom_predict(model, &label, &prob);

            // output
            if(prob > 0.9f) {
                printf("%s : %d%%\n", (char*)&label_name[label], (int)(prob * 100));
            }
        }
    }

    if (dev) {
        snd_pcm_drain(dev);
        snd_pcm_close(dev);
    }

    if( mfcc != NULL ) {
        mfcc_delete(mfcc);
    }

    printf("== kws exit\n");
}
