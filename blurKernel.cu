#include	"wb.h"


void wbImage_save(const wbImage_t& image, const char* fName){
	std::ostringstream oss;
        oss << "P6\n" << "# Created for blurring output" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";
	//oss << "P6\n" << "# Created by GIMP version 2.10.8 PNM plug-in" << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";

        std::string headerStr(oss.str());

	std::ofstream outFile(fName, std::ios::binary);
        outFile.write(headerStr.c_str(), headerStr.size());

        const int numElements = image.width * image.height * image.channels;

        unsigned char* rawData = new unsigned char[numElements];

        for (int i = 0; i < numElements; ++i)
        {
            rawData[i] = static_cast<unsigned char>(image.data[i] * wbInternal::kImageColorLimit + 0.5f);
        }

        outFile.write(reinterpret_cast<char*>(rawData), numElements);
        outFile.close();

        delete [] rawData;
}

#define BLOCK_SIZE 16
#define Mask_width  4
#define Mask_height 4
#define Mask_radius_x Mask_width/2
#define Mask_radius_y Mask_height/2
#define TILE_WIDTH 16
#define w_x (TILE_WIDTH + Mask_width - 1) //Shared Memory Elements needed to load as per Mask Size
#define w_y (TILE_WIDTH + Mask_height - 1)


__device__ __constant__ int BLUR_SIZE = 1;

__global__
void blurKernel(float *in, float *out, int width, int height,int imageChannels){

	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(Col < width && Row < height){
		float pixValRed = 0;
		float pixValGreen = 0;
		float pixValBlue = 0;
		int pixels = 0;

		//Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box

		for(int blurRow = -BLUR_SIZE - 1; blurRow < BLUR_SIZE + 1; ++blurRow){
			for(int blurCol = -BLUR_SIZE - 1; blurCol < BLUR_SIZE + 1; ++blurCol){

			int currentRow = Row + blurRow;
			int currentCol = Col + blurCol;

			if(currentRow > -1 && currentRow < height && currentCol > -1 && currentCol < width){
				int rgbOffset = imageChannels * (currentRow * width + currentCol);
				pixValRed += in[rgbOffset];
				pixValGreen += in[rgbOffset + 1];
				pixValBlue += in[rgbOffset + 2];
				pixels++; //Keep track of number of pixels in the avg
			
			}
			
                        					

		}
	}
	//Write our new pixel value out			
	out[imageChannels * (Row * width + Col)] = (float)(pixValRed / pixels);
	out[imageChannels * (Row * width + Col) + 1] = (float)(pixValGreen / pixels);
	out[imageChannels * (Row * width + Col) + 2] = (float)(pixValBlue / pixels);
	
	
	}
	
}



__global__ void blurKernelShared(float *in, float *out, int width, int height, int imageChannels)
{
   __shared__ float N_ds[w_y][w_x];
   int k;
   for (k = 0; k < imageChannels; k++)
   {
    //destY & destX: Row and Column of Shared Memory
    //srcY & srcX: Indexes to fetch data from input Image
    //src: index of Input Image

    //loading
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
         destY = dest / w_x, destX = dest % w_x,
         srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius_x,
         srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius_y,
         src = (srcY * width + srcX) * imageChannels + k;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY][destX] = in[src];
      else
         N_ds[destY][destX] = 0.0;

        for (int iter=1; iter <= (w_x * w_y) / (TILE_WIDTH*TILE_WIDTH); iter++)
        {
          // Second loading
          dest = threadIdx.y * TILE_WIDTH + threadIdx.x + iter*(TILE_WIDTH * TILE_WIDTH);
            destY = dest / w_x, destX = dest % w_x;
            srcY  = blockIdx.y * TILE_WIDTH + destY - Mask_radius_x;
            srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius_y;
            src = (srcY * width + srcX) * imageChannels + k;
            if (destY < w_y && destX < w_x)
            {
                if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                    N_ds[destY][destX] = in[src];
                else
                    N_ds[destY][destX] = 0.0;
            }
        }
      __syncthreads();

      float pixVal = 0;
      int y, x;
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            pixVal += N_ds[threadIdx.y + y][threadIdx.x + x] / 16;
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
       out[(y * width + x) * imageChannels + k] = pixVal;
      __syncthreads();
   }
}
	


int main(int argc, char ** argv) {
    
    char * inputImageFile;
    char * outputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;

    float * hostInputImageData;
    float * hostOutputImageData;

    inputImageFile = argv[1];
    outputImageFile = argv[2];
    printf("Loading %s\n", inputImageFile);
    inputImage = wbImport(inputImageFile);
    hostInputImageData = wbImage_getData(inputImage);

    int imageWidth = wbImage_getWidth(inputImage);
    int imageHeight = wbImage_getHeight(inputImage);
    int imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth,imageHeight,imageChannels);
    hostOutputImageData = wbImage_getData(outputImage);

    hostInputImageData = wbImage_getData(inputImage);

    //Define size 
    size_t size = imageHeight * imageWidth * imageChannels * sizeof(float);
   
    printf("%d %d %d\n", imageWidth, imageHeight, imageChannels);
    printf("%f %f %f\n", hostInputImageData[0], hostInputImageData[1], hostInputImageData[2]);

 
    /*YOUR CODE HERE*/

    //Define error variable
    cudaError_t err = cudaSuccess;

    float * deviceInputImageData;
    float * deviceOutputImageData;
    //Allocate device memory for copy
    err = cudaMalloc((void **)&deviceInputImageData, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector X (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&deviceOutputImageData, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector X (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   //Copy vector from host to device
    printf("Copy input image data from the host memory to the CUDA device\n");
    err = cudaMemcpy(deviceInputImageData, hostInputImageData, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector X from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(deviceOutputImageData, hostOutputImageData, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector Y from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Define Cuda Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Launch the Cuda Kernel

    dim3 dimGrid(ceil(imageWidth / 16.0), ceil(imageHeight / 16.0), 1);
    dim3 dimBlock(16,16, 1);
    
    cudaEventRecord(start);
    blurKernel<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight, imageChannels);
    cudaEventRecord(stop);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy result vector from device to host
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(hostOutputImageData, deviceOutputImageData, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Timer : %lf\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    outputImage.data = hostOutputImageData;
    wbImage_save(outputImage, outputImageFile); 

    //free variables

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
  
    printf("DONE !");
    return 0;
}


 

