
#include <cuda_runtime.h>

#define min(a, b)  ((a) < (b) ? (a) : (b))
#define num_threads   512

typedef unsigned char uint8_t;

struct Size{  // 直接include opencv 也是可以的
    int width = 0, height = 0;

    Size() = default;
    Size(int w, int h)
    :width(w), height(h){}  // 输入w 更新初始化的值
};

// 计算仿射变换矩阵
// 计算的矩阵是居中缩放
struct AffineMatrix{
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr5UdL/
     */
    // 这边就是建议从image到dst 和从dst到image的变换矩阵给存住
    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    // 这里其实是求解imat的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    // 从哪   到哪  也就是原始尺寸到目标尺寸
    void compute(const Size& from, const Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        // 这里取min的理由是
        // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
        // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
        // **
        float scale = min(scale_x, scale_y); // 缩放比例辅助视频讲解 https://v.douyin.com/NhrH8Gm/
        // 为什么取小的比例，看这个视频
        // 记住吧
        /**
        这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
        scale, 0, -scale * from.width * 0.5 + to.width * 0.5
        0, scale, -scale * from.height * 0.5 + to.height * 0.5
        
        这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
        例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
        S = [
        scale,     0,      0
        0,     scale,      0
        0,         0,      1
        ]
        
        P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
        P = [
        1,        0,      -scale * from.width * 0.5
        0,        1,      -scale * from.height * 0.5
        0,        0,                1
        ]

        T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
        T = [
        1,        0,      to.width * 0.5,
        0,        1,      to.height * 0.5,
        0,        0,            1
        ]

        通过将3个矩阵顺序乘起来，即可得到下面的表达式：
        M = [
        scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
        0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
        0,        0,                     1
        ]
        去掉第三行就得到opencv需要的输入2x3矩阵
        **/

        /* 
            + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
            参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        // 前两行
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 
            -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;  // 这边多减了个0.5

        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 
            -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
        // 逆矩阵，写一个方法
        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){

    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height,
    float* dst, int dst_line_size, int dst_width, int dst_height,
	uint8_t fill_value, AffineMatrix matrix
){
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr4vTF/
     */
    // 进行了局部的idx->全部idx的转换   因为想让一个thread算一个像素（也就是三个通道的）
    // 作者说的就是每一个block里面的每一个thread都是去处理每一像素对应的三个通道的值
    int dx = blockDim.x * blockIdx.x + threadIdx.x;  // 1D layout
    int dy = blockDim.y * blockIdx.y + threadIdx.y;  // 1D layout
    // 作者和老师讲的还不一样，什么全局的坐标之前的13举例子，全局的二维的坐标说是什么（1，5），他是希望转成这种二维的？
    if (dx >= dst_width || dy >= dst_height)  return;  // 线程数超过了图像大小了   完全放到二维理解？

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    float src_x = 0; float src_y = 0;
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);  // 拿到目标图的xy的坐标，映射成输入图的xy的坐标，通过matrix映射
	// 双线性插值，此时dx dy 已经和原始的能映射对应起来了，然后操作原始的？ 最后得到我实际目标的dx dy的RGB三个像素的值是多少
    
    /*
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - 双线性理论讲解：https://v.douyin.com/NhrH2tb/
        - 代码代码：https://v.douyin.com/NhrBqpc/ 
     */
    // 下面的就是出现了四个格子，有些格子直接跑到了原本的像素的外面了怎么办
    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
    }else{
        int y_low = floorf(src_y);  // （x_low,y_low）左上角  去整
        int x_low = floorf(src_x);  // （srcx和srcy）是从目标图像的像素逆映射回来的亚像素的感觉
        int y_high = y_low + 1;  // （x_high,y_high） 右下角
        int x_high = x_low + 1;
        // 超出边界的点该给什么值，因为是三通道的，就需要给三个值
        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        float ly    = src_y - y_low;  // （srcx和srcy） 距离四个边的值
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;  // 四个区域的面积  注意顺序
        //w4  w3     正好是反过来的 1 2
        //w1  w2                 4 3     因为 看图的话，实际上我离这个点越近，实际上我的权重是越大的，但是我的面积就得更大，所以是对角的那种感觉
        uint8_t* v1 = const_values;  // 四个最邻近的像素点的值，先赋予一个初值
        uint8_t* v2 = const_values;
        uint8_t* v3 = const_values;
        uint8_t* v4 = const_values;
        // 如果V1 - V4没取到值，那么就是常数值，如果四个值都在原始图像内就正常算，如果出现只有一个点在里面，其余三个点都跑到外面去了，那么就需要把其余的三个点置为常数
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;  // 看抖音 这些什么x_low * 3  x_high * 3 就可以想象成：内存一长条，实际v1-v4是连续的
                // 连续的内存空间  这个大的分支是用y_low来填充  而下面的是根据y_high 来填充
                // 所以说v1 v2 在上面都是y low    同理v1 v3都是x low   v2 v4 都是x high  就是左上角右下角，内存乘了一长条那种，个人理解不知道对不对
                //      v3 v4 在下面都是y high

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        //v1【0】【1】【2】 就是RGB三个值，w应该都是小于1的
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }
	// 
    // 所以下面就和上面是一致的，都是那种内存展开成一长条那种感觉
    // uint8_t* pdst = dst + dy * dst_line_size + dx * 3; // 这个干啥的，pdst代表当前处理的像素的地址是什么，就是目标图的像素的首地址，做了横向和纵向的偏移
    // 关于纵向的偏移，就是dy * 一行的长度，dst_line_size 传参数的时候就等于image.cols * 3 这样的，所以也是三倍，（一行多少像素 * 3）
    // 关于横向的dx*3 就是每一个像素都是RGB三个值需要处理；  但是这样子的话，是每三个一取还是三个三个一取？ 后面按照0-1-2索引填的
    // **************所以这个dst 就是当前这个线程的某一小块的首地址，是吧**************
    // pdst[0] = c0;  //R
    // pdst[1] = c1;   // G
    // pdst[2] = c2;  // 有了每个像素的地址之后，就可以填入变换后的值了
	// BGR->RGB:  顺序交换   pdst[2] = c0; pdst[0] = c1; pdst[1] = c2;     因为我在操作像素，只要一行代码就能实现之前的操作一个函数的功能，效率极高
	// 减均值除方差：
	// transpose：totensor   索引值的偏移量计算下降就行了，后面有
    //bgr to rgb
    float t = c2;
    c2 = c0;
    c0 = t;

    //normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    //rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void warp_affine_bilinear(
    /*
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhre7fV/
     */
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_line_size, int dst_width, int dst_height,
	uint8_t fill_value
){
    dim3 block_size(32, 32); // blocksize最大就是1024，这里用2d来看更好理解  2d layout  就是两个1D layout  blocksize最大1024  32*32=1024，不能超过的xyz乘起来
    dim3 grid_size((dst_width + 31) / 32, (dst_height + 31) / 32);  //典型的向上取整的操作  640+31 /32   +blocksize - 1/blocksize 骚操作
    // 为什么用dst的，是因为用什么反向的mapping
    AffineMatrix affine;  // 仿射矩阵  结构体
    affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));  // 直接算出矩阵
	// 看着layout很复杂，实际上可以看成这个启动的线程数就是图像的宽度乘以图像的高度，目标图像的高度和大小， dist，不能整除32就会多取一点，线程数就会多一点，在核函数里面直接过滤掉多出来的就行了
    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>(
        src, src_line_size, src_width, src_height, // 这边的参数 和传进来的类似
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine
    );
}
