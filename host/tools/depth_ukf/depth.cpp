/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#include <string.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <math.h>

#include "misc.h"
#include "image.h"
#include "pnmfile.h"
#include "filter.h"
#include "imconv.h"

//#include "cv.h"

#define ITER 10       // number of BP iterations at each scale
#define LEVELS 5     // number of scales

#define DISC_K 55.0F         // truncation of discontinuity cost
#define DATA_K 1000000.0F        // truncation of data cost
#define LAMBDA 1.0F        // weighting of data cost

#define INF 1E20     // large cost
#define VALUES 100   // number of possible depths
#define SCALE 1     // scaling from disparity to graylevel in output

#define SIGMA 0.05    // amount to smooth the input images

#define MASK_DIMEN 101

#define NUM_OF_TRANS 129


double costFunc( image<uchar>* G, image<uchar>* F, image<float>* H);
image<float[VALUES]> *comp_data(image<uchar> *img1, image<uchar> *img2, float *tx,float *ty, float *weight);


// dt of 1d function
static void dt(float f[VALUES]) {
  for (int q = 1; q < VALUES; q++) {
    float prev = f[q-1] + 1.0F;
    if (prev < f[q])
      f[q] = prev;
  }
  for (int q = VALUES-2; q >= 0; q--) {
    float prev = f[q+1] + 1.0F;
    if (prev < f[q])
      f[q] = prev;
  }
}

// compute message
void msg(float s1[VALUES], float s2[VALUES],
    float s3[VALUES], float s4[VALUES],
    float dst[VALUES]) {
  float val;

  // aggregate and find min
  float minimum = INF;
  for (int value = 0; value < VALUES; value++) {
    dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
    if (dst[value] < minimum)
      minimum = dst[value];
  }

  // dt
  dt(dst);

  // truncate 
  minimum += DISC_K;
  for (int value = 0; value < VALUES; value++)
    if (minimum < dst[value])
      dst[value] = minimum;

  // normalize
  val = 0;
  for (int value = 0; value < VALUES; value++)
    val += dst[value];

  val /= VALUES;
  for (int value = 0; value < VALUES; value++)
    dst[value] -= val;
}

// generate output from current messages
image<uchar> *output(image<float[VALUES]> *u, image<float[VALUES]> *d, 
    image<float[VALUES]> *l, image<float[VALUES]> *r, 
    image<float[VALUES]> *data)
{
  int width = data->width();
  int height = data->height();
  image<uchar> *out = new image<uchar>(width, height);

  for (int y = 1; y < height-1; y++) {
    for (int x = 1; x < width-1; x++) {
      // keep track of best value for current pixel
      int best = 0;
      float best_val = INF;
      for (int value = 0; value < VALUES; value++) {
        float val = 
          imRef(u, x, y+1)[value] +
          imRef(d, x, y-1)[value] +
          imRef(l, x+1, y)[value] +
          imRef(r, x-1, y)[value] +
          imRef(data, x, y)[value];
        if (val < best_val) {
          best_val = val;
          best = value;
        }
      }
      imRef(out, x, y) = best * SCALE;
    }
  }

  return out;
}

// belief propagation using checkerboard update scheme
void bp_cb(image<float[VALUES]> *u, image<float[VALUES]> *d,
    image<float[VALUES]> *l, image<float[VALUES]> *r,
    image<float[VALUES]> *data,
    int iter) {
  int width = data->width();  
  int height = data->height();

  for (int t = 0; t < ITER; t++) {
    std::cout << "iter " << t << "\n";

    for (int y = 1; y < height-1; y++) {
      for (int x = ((y+t) % 2) + 1; x < width-1; x+=2) {

        msg(imRef(u, x, y+1),imRef(l, x+1, y),imRef(r, x-1, y),
            imRef(data, x, y), imRef(u, x, y));

        msg(imRef(d, x, y-1),imRef(l, x+1, y),imRef(r, x-1, y),
            imRef(data, x, y), imRef(d, x, y));

        msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(r, x-1, y),
            imRef(data, x, y), imRef(r, x, y));

        msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(l, x+1, y),
            imRef(data, x, y), imRef(l, x, y));

      }
    }
  }
}

// multiscale belief propagation for image restoration
image<uchar> *depth_ms(image<uchar> *img1, image<uchar> *img2,float *tx,float *ty,float *weight) {
  image<float[VALUES]> *u[LEVELS];
  image<float[VALUES]> *d[LEVELS];
  image<float[VALUES]> *l[LEVELS];
  image<float[VALUES]> *r[LEVELS];
  image<float[VALUES]> *data[LEVELS];

  printf("Entered depth ms function\n");
  //image<float>* h = new image<float> ( 100, 100);

  // data costs
   data[0] = comp_data(img1,img2,tx,ty,weight);

  // data pyramid
  for (int i = 1; i < LEVELS; i++) {
    int old_width = data[i-1]->width();
    int old_height = data[i-1]->height();
    int new_width = (int)ceil(old_width/2.0);
    int new_height = (int)ceil(old_height/2.0);

    assert(new_width >= 1);
    assert(new_height >= 1);

    data[i] = new image<float[VALUES]>(new_width, new_height);
    for (int y = 0; y < old_height; y++) {
      for (int x = 0; x < old_width; x++) {
        for (int value = 0; value < VALUES; value++) {
          imRef(data[i], x/2, y/2)[value] += imRef(data[i-1], x, y)[value];
        }
      }
    }
  }

  // run bp from coarse to fine
  for (int i = LEVELS-1; i >= 0; i--) {
    int width = data[i]->width();
    int height = data[i]->height();

    // allocate & init memory for messages
    if (i == LEVELS-1) {
      // in the coarsest level messages are initialized to zero
      u[i] = new image<float[VALUES]>(width, height);
      d[i] = new image<float[VALUES]>(width, height);
      l[i] = new image<float[VALUES]>(width, height);
      r[i] = new image<float[VALUES]>(width, height);
    } else {
      // initialize messages from values of previous level
      u[i] = new image<float[VALUES]>(width, height, false);
      d[i] = new image<float[VALUES]>(width, height, false);
      l[i] = new image<float[VALUES]>(width, height, false);
      r[i] = new image<float[VALUES]>(width, height, false);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          for (int value = 0; value < VALUES; value++) {
            imRef(u[i], x, y)[value] = imRef(u[i+1], x/2, y/2)[value];
            imRef(d[i], x, y)[value] = imRef(d[i+1], x/2, y/2)[value];
            imRef(l[i], x, y)[value] = imRef(l[i+1], x/2, y/2)[value];
            imRef(r[i], x, y)[value] = imRef(r[i+1], x/2, y/2)[value];
          }
        }
      }      
      // delete old messages and data
      delete u[i+1];
      delete d[i+1];
      delete l[i+1];
      delete r[i+1];
      delete data[i+1];
    } 

    // BP
    bp_cb(u[i], d[i], l[i], r[i], data[i], ITER);
  }

  image<uchar> *out = output(u[0], d[0], l[0], r[0], data[0]);

  delete u[0];
  delete d[0];
  delete l[0];
  delete r[0];
  delete data[0];

  return out;
}

image<float> *translate(image<uchar> *img1, float tx, float ty)
{

   float src_x,src_y;
   float delta_x,delta_y;
   int height,width;
   int top,bottom,left,right;

   width = img1->width();
   height = img1->height();
   

   image<float>* out = new image<float>(img1->width(),img1->height());
   image<float> *sm1;
   sm1 = imageUCHARtoFLOAT(img1);

   for(int i = 0;i<img1->height();i++)
     for(int j = 0;j<img1->width();j++)
     {
         src_x = j - tx;
         src_y = i - ty;
         top = floor(src_y);
         bottom = ceil(src_y);
         left = floor(src_x);
         right = ceil(src_x);

         delta_x = src_x - floor(src_x);
         delta_y = src_y - floor(src_y);

        
         if(src_x > 0 && src_x < (img1->width()-1) && src_y > 0 && src_y < (img1->height()-1))
         {
              out->data[i*width + j] =  (((1-delta_x)*(1-delta_y)*sm1->data[top*width + left]) + ((delta_x)*(1-delta_y)*sm1->data[top*width + right]) + ((1-         delta_x)*(delta_y)*sm1->data[bottom*width + left]) + ((delta_x)*(delta_y)*sm1->data[bottom*width + right]));                       
         }
     }
  
   delete sm1;
   return out;

}


image<float> *blur_img_gen(image<uchar> *img,float *tx,float *ty,float *weight)
{

  image<float> *temp;
  image<float> *out = new image<float> (img->width(),img->height());
  int height,width;

  height = img->height();
  width = img->width();


  // Accumulating the result for various transformations
  for(int i=0;i<NUM_OF_TRANS;i++)
  {
     temp = translate(img,tx[i],ty[i]);
   //  printf("%f\n",weight[i]);
     for(int l = 0;l<height;l++)
        for(int m = 0;m<width;m++)
        {
           out->data[l*width + m] = out->data[l*width + m] + weight[i]*temp->data[l*width + m];
        }

  }

  
  return out;
 
}

int main(int argc, char **argv) {
  image<uchar> *img1, *img2;
  image<uchar> *out;
  float *tx,*ty,*weight; 

  tx = (float *)calloc(NUM_OF_TRANS,sizeof(float));
  ty = (float *)calloc(NUM_OF_TRANS,sizeof(float));
  weight = (float *)calloc(NUM_OF_TRANS,sizeof(float));

  int i,j;

  if (argc != 6) {
    std::cerr << "usage: " << argv[0] << " tx(txt) ty(txt) left(pgm) right(pgm) out(pgm)\n";
    exit(1);
  }

  // load input
  img1 = loadPGM(argv[3]);
  img2 = loadPGM(argv[4]);
  out = new image<uchar> (img1->width(),img1->height());

  int height,width;

  height = img1->height();
  width = img1->width();
  
  //Load the transformations
  FILE* fp1 = fopen(argv[1],"r");
  FILE* fp2 = fopen(argv[2],"r");
  float temp;
  for(i=0;i<NUM_OF_TRANS;i++)
  {
    fscanf(fp1,"%f",&temp);
    tx[i] = temp;
    fscanf(fp2,"%f",&temp);
    ty[i] = temp;
    weight[i] = 1.0/(float)NUM_OF_TRANS;
  }

  fclose(fp1);
  fclose(fp2);
  

  out = depth_ms(img1,img2,tx,ty,weight);

  // save output
  savePGM(out, argv[5]);

  delete img1;
  delete img2;
  delete out;
  delete tx;
  delete ty;
  return 0;
}



image<float[VALUES]> *comp_data(image<uchar> *img1, image<uchar> *img2, float *tx, float *ty,float *weight) {
 
  //Defining the array to store the data cost
  int width,height;
  width = img1->width();
  height = img1->height();

  image<float[VALUES]> *data = new image<float[VALUES]>(width, height);

  float *scaled_tx, *scaled_ty;

  scaled_tx = (float*)calloc(NUM_OF_TRANS,sizeof(float));
  scaled_ty = (float*)calloc(NUM_OF_TRANS,sizeof(float));
  float scale;
    
  image<float> *sm1, *sm2;
   scale = 0;
    for(int i=0;i<VALUES;i++)
   {
/*      float kar_temp1 = (float)VALUES;
      float kar_temp2 = (float)(i+1);
      scale = kar_temp2/kar_temp1;*/
     
      
      printf("scale  = %f\n",scale);    
      for(int l = 0;l<NUM_OF_TRANS;l++)
      {
          scaled_tx[l] = tx[l]*scale;
          scaled_ty[l] = ty[l]*scale;
      }     
      
      printf("labels = %d\n",i);       
      // Generation Of blurred image as per the scaled transformations
      sm2 = blur_img_gen(img1,scaled_tx,scaled_ty,weight);

      for (int y = 0; y < height; y++) {
       for (int x = 0; x < width; x++) {
	float val = abs(imRef(img2, x, y)-imRef(sm2, x, y));	
//	imRef(data, x, y)[i] = LAMBDA * std::min(val, DATA_K);
        imRef(data, x, y)[i] = LAMBDA * val;
       }
      }

      scale = scale + 0.01;
   }

      printf("Data cost is calculated\n");
      delete sm1;
      delete sm2;
      return data;
}

