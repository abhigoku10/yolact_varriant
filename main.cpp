#include <opencv2/opencv.hpp>
#include <iostream>
#include <json/json.h>
#include <fstream>
#include "NumCpp.hpp"
#include <numeric>
#include <vector>
#include <cmath>

// using namespace cv;
// using namespace nc;
using namespace std;

cv::Mat readImage(cv::String path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    return img;
}

void saveImage(cv::Mat image, cv::String path)
{
    cv::imwrite(path, image);
}


Json::Value readJson(cv::String path)
{
    Json::Value root;
    std::ifstream ifs;
    ifs.open(path);

    Json::CharReaderBuilder builder;
    builder["collectComments"] = true;
    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &root, &errs)) 
    {
        std::cout << errs << std::endl;
        return EXIT_FAILURE;
    }    
    return root;
}

void showImage(cv::Mat image)
{
    cv::String windowName = "The Guitar"; //Name of the window
    cv::namedWindow(windowName); // Create a window
    cv::imshow(windowName, image); // Show our image inside the created window.
    cv::waitKey(0); // Wait for any keystroke in the window
    cv::destroyWindow(windowName); //destroy the created window
}

void getAdjoiningPoints(nc::NdArray<int> &orderedPoints, nc::NdArray<int> intersectEdges)
{
    // adjPoints = intersectingEdge[(intersectingEdge[:, 0] > (basePoints[-1, 0] - 2))\
    //     * (intersectingEdge[:, 0] < (basePoints[-1, 0] + 2))\
    //     * (intersectingEdge[:, 1] > (basePoints[-1, 1] - 2))\
    //     * * (intersectingEdge[:, 1] < (basePoints[-1, 1] + 2))]
    for (unsigned i = 0; i < intersectEdges.size(); i += 2)
        if(intersectEdges[i] > (orderedPoints[-2] - 2))
            if(intersectEdges[i] < (orderedPoints[-2] + 2))            
                if(intersectEdges[i + 1] > (orderedPoints[-1] - 2))
                    if(intersectEdges[i + 1] < (orderedPoints[-1] + 2))
                    {
                        orderedPoints = nc::append(orderedPoints, {int(intersectEdges[i]), int(intersectEdges[i])}, nc::Axis::ROW);
                        break;
                    }
}

void edgeParams(nc::NdArray<nc::uint8> image, unsigned rows, unsigned cols, nc::NdArray<int> &openEdges, nc::NdArray<int> &intersectEdges)
{
    
    auto a1 = nc::nonzero(image);
    int count = 0;
    for(unsigned i = 0; i < a1.first.numCols(); i++)
    {
        // Calculate surrounding sum to determine open pixel values
        if (a1.first[i] > 0 && a1.second[i] > 0 && a1.first[i] < rows && a1.second[i] < cols)
        {
            int top = a1.first[i] - 1;
            int left = a1.second[i] - 1;
            int bottom = a1.first[i] + 2;
            int right = a1.second[i] + 2;
            auto sum = nc::sum(image({top, bottom}, {left, right}).astype<nc::int32>());
            if (sum[0] <= 511)
            {   
                count += 1;
                openEdges = nc::append(openEdges, {int(a1.first[i]), int(a1.second[i])});
            }
            else if (sum[0] >= 1020)
            {
                intersectEdges = nc::append(intersectEdges, {int(a1.first[i]), int(a1.second[i])});

            }
        }
    }
}

nc::NdArray<int> getComponents(nc::NdArray<nc::uint8> image, nc::NdArray<int> intersectEdges, nc::NdArray<int> &indexes, unsigned &labelCount)
{
    cv::Mat stats, centroids, nz, labels;    
    
    for (unsigned i = 0; i < intersectEdges.size(); i += 2)
        image(intersectEdges[i], intersectEdges[i + 1]) = nc::uint8{0};
    cv::Mat cvArray = cv::Mat(image.numRows(), image.numCols(), CV_8U, image.data());
    labelCount = cv::connectedComponentsWithStats(cvArray, labels, stats, centroids);
    nc::NdArray<int> componentPoints = nc::NdArray<int>({0,0});
    cv::findNonZero(labels, nz);
    for(unsigned j=1; j< labelCount; j++)
    {
        for(unsigned i=0; i< nz.total(); i++)
        {
            if(labels.at<int>(nz.at<cv::Point>(i).y, nz.at<cv::Point>(i).x) == j)
                componentPoints = nc::append(componentPoints,  {int(nz.at<cv::Point>(i).y), int(nz.at<cv::Point>(i).x)});
        }            
        indexes = nc::append(indexes, {int(componentPoints.size())});
    }
        
                
    return componentPoints;
    
}

nc::NdArray<nc::uint8> L2PadImageByOne(nc::NdArray<nc::uint8> image)
{
    auto padImage = nc::zeros<nc::uint8>(image.numRows() + 2, image.numCols() + 2);
    auto nz = nc::nonzero(image);
    for(unsigned i = 0; i < nz.first.numCols(); i++)
        padImage(nz.first[i] + 1, nz.second[i] + 1) = image(nz.first[i], nz.second[i]);
    return padImage;
}

nc::NdArray<nc::uint8> diskStructureKernel(int radius)
{
    auto fillValue = nc::uint8{1};
    auto y = nc::arange(-1*radius, radius + 1);
    auto x = nc::arange(-1*radius, radius + 1);
    auto kernel = nc::zeros<nc::uint8>((radius * 2) + 1, (radius * 2) + 1);
    auto cols = int(kernel.numCols());
    auto rows = int(kernel.numRows());

    for(unsigned i = 0; i < y.size(); i++)
        for(unsigned j = 0; j < x.size(); j++)
            if ( (pow(x[j],2) + pow(y[i],2)) <= pow(radius,2) )
                kernel(i, j) = fillValue;
    
    for(int i = radius -1; i< cols - radius + 1; i++)
    {
        kernel(0,i) = fillValue;
        kernel(rows-1,i) = fillValue;
        kernel(i,0) = fillValue;
        kernel(i,rows - 1) = fillValue; 
    }

    return kernel;
}

nc::NdArray<nc::uint8> morphology(nc::NdArray<nc::uint8> image, nc::NdArray<nc::uint8> kernel)
{
    cv::Mat tempImage = cv::Mat(image.numRows(), image.numCols(), CV_8U, image.data());
    cv::Mat tempKernel = cv::Mat(kernel.numRows(), kernel.numCols(), CV_8U, kernel.data());
    cv::Mat output;
    cv::morphologyEx(tempImage, output, cv::MORPH_OPEN, tempKernel); 
    // saveImage(output, "morph.png");
    return nc::NdArray<nc::uint8>(output.data, output.rows, output.cols); 
}

float euclidDistance(int x1, int y1, int x2, int y2)
{
    return float(pow((pow((y1 - y2), 2) + pow((x1 - x2), 2)), 0.5));
}

nc::NdArray<int> L3SortSKCoordPts(nc::NdArray<int> points, cv::Point2i endPoint)
{
    // If endPoint Specified
    nc::NdArray<int> sortedPoints = {{endPoint.y, endPoint.x}};
    unsigned indexMinDist;
    float distance;
    for(unsigned j=0; j < points.size(); j+=2)
    {
        float minDist = 999.0;
        for(unsigned i=0; i < points.size(); i+=2)
        {
            if(int(points[i]) == -1)
                continue;
            distance = euclidDistance(points[i+1], points[i], endPoint.x, endPoint.y);
            if (distance < minDist)
            {
                minDist = distance;
                indexMinDist = i;
            }
        }
        endPoint.y = int(points[indexMinDist]);
        endPoint.x = int(points[indexMinDist+1]);
        if(j != 0)
            sortedPoints = nc::append(sortedPoints, {int(points[indexMinDist]), int(points[indexMinDist + 1])}, nc::Axis::ROW);
        points[indexMinDist] = -1;
        points[indexMinDist + 1] = -1;
    }
    return sortedPoints;
}

nc::NdArray<int> getCckWiseNeighborPixVals(int y, int x, nc::NdArray<nc::uint8> image)
{
    auto clockwisePoints = nc::zeros<int>(8, 1);
    clockwisePoints(0, 0) = int(image(y - 1, x));
    clockwisePoints(1, 0) = int(image(y - 1, x + 1));
    clockwisePoints(2, 0) = int(image(y, x + 1));
    clockwisePoints(3, 0) = int(image(y + 1, x + 1));
    clockwisePoints(4, 0) = int(image(y + 1, x));
    clockwisePoints(5, 0) = int(image(y + 1, x - 1));
    clockwisePoints(6, 0) = int(image(y, x - 1));
    clockwisePoints(7, 0) = int(image(y - 1, x - 1));

    return clockwisePoints;
}

int numTransitions(nc::NdArray<int> clockwisePoints, int p1, int p2)
{
    auto pointPair = clockwisePoints;
    int sum = 0;
    for(unsigned i = 0; i < clockwisePoints.size(); i+=2)
    {
        pointPair(i, 0) += clockwisePoints(0, 0);
        pointPair(i + 1, 0) += clockwisePoints(0, 0);
        if (pointPair(i, 0) == p1)
            if (pointPair(i + 1, 0) == p2)
                sum += 1;
    }
    return sum;
}

nc::NdArray<int> L3SortSKCoordPts(nc::NdArray<nc::uint8> image)
{
    cv::Point2i endPoint;
    auto a1 = nc::nonzero(image);
    auto nz = nc::empty<int>(a1.first.numCols(), 2);
    for(unsigned i = 0; i < a1.first.numCols(); i++)
    {
        nz(i, 0) = a1.first[i];
        nz(i, 1) = a1.second[i];
    }
    for(unsigned i = 0; i < a1.first.numCols(); i++)
    {
        auto clockwisePoints = getCckWiseNeighborPixVals(int(a1.first[i]), int(a1.second[i]), image);
        auto arraySum = nc::sum(clockwisePoints);
        if (arraySum[0] == 255) 
        {
            endPoint.y = a1.first[i];
            endPoint.x = a1.second[i];
            break;
        }
        else if(arraySum[0] == 2*255)
        {
            auto sum = numTransitions(clockwisePoints, 255, 255);
            if (sum == 1)
            {
                endPoint.y = a1.first[i];
                endPoint.x = a1.second[i];
                break;
            }
        }
    }
    return L3SortSKCoordPts(nz, endPoint);
}

nc::NdArray<nc::uint8> skeletonize(nc::NdArray<nc::uint8> img)
{
    int arr[] = {0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0};
    img = img / nc::uint8{255};
    auto padImage = L2PadImageByOne(img);
    auto skeletonImage = padImage;
    int pixRemoved = 1;
    int count = 0;
    while (pixRemoved == 1)
    {
        pixRemoved = 0;
        for(int k=0; k< 2; k++)
        {
            for(unsigned i = 1; i < (padImage.numRows() - 1); i++)
                for(unsigned j = 1; j < (padImage.numCols() - 1); j++)
                {
                    if(skeletonImage(i, j) != 0)
                    {
                        auto neighbors = skeletonImage(i - 1, j - 1) + 2*skeletonImage(i - 1, j) +\
                            4*skeletonImage(i - 1, j + 1) + 8*skeletonImage(i, j + 1) +\
                            16*skeletonImage(i + 1, j + 1) + 32*skeletonImage(i + 1, j) +\
                            64*skeletonImage(i + 1, j - 1) + 128*skeletonImage(i, j - 1);
                        if(((arr[neighbors] == 1) && (k == 0)) ||\
                            ((arr[neighbors] == 2) && (k != 0)) ||\
                            (arr[neighbors] == 3))
                        {
                            padImage(i, j) = 0;
                            pixRemoved = 1;
                        }            
                    }
                }
            skeletonImage = padImage;
        }
    }
    skeletonImage = skeletonImage * nc::uint8{255};
    return skeletonImage({1,int(padImage.numRows() - 1)}, {1,int(padImage.numCols() - 1)});
}


nc::NdArray<double> L3Check2BreakNecks(nc::NdArray<nc::uint8> objMask, nc::NdArray<int> orderedPoints,  double normY, double normX, int neckLengthTh, float scanWFactor, unsigned &truth)
{
            
    auto gradV = nc::NdArray<double>{{normX*scanWFactor, -1 * normY * scanWFactor}};
    auto scanNeigh = nc::NdArray<double>{{double(orderedPoints(0, 0)), double(orderedPoints(0, 1))},\
                                            {double(orderedPoints(0, 0) + gradV(0,0)), double(orderedPoints(0, 1) + gradV(0,1))},\
                                            {double(orderedPoints(0, 0) - gradV(0,0)), double(orderedPoints(0, 1) - gradV(0,1))}};
    double EPSILON = 0.001;
    auto initiate0 = orderedPoints(0,0) + normY;
    auto initiate1 = orderedPoints(0,1) + normX;
    initiate0 = max(1+EPSILON, initiate0);
    initiate1 = max(1+EPSILON, initiate1);
    auto scanPt = nc::NdArray<double>{{min(initiate0, double(objMask.numRows() - 2 - EPSILON)),\
                                        min(initiate1, double(objMask.numCols() - 2 - EPSILON))}};
    // scanPt.print();
    // scanNeigh = nc::append(scanNeigh, {double(orderedPoints(0, norm*2) - gradV(0,0)), double(orderedPoints(0, norm*2 + 1) - gradV(0,0))}, nc::Axis::ROW);
    double minDist2ZeroP = 10000.0;
    for(int k = 0; k<int(neckLengthTh /2 + 0.5); k++)
    {
        if(objMask(int(scanPt(0,0)), int(scanPt(0,1))) == nc::uint8{0})
        {
            auto zero2RefDist = pow((int(scanPt(0,0)) - int(orderedPoints(0, 0))), 2) + pow((int(scanPt(0,1)) - double(orderedPoints(0, 1))), 2);
            minDist2ZeroP = min(zero2RefDist, minDist2ZeroP);
        }
        if(objMask(int(scanPt(0,0) + gradV(0,0)), int(scanPt(0,1) + gradV(0,1))) == nc::uint8{0})
        {
            auto zero2RefDist = pow((int(scanPt(0,0) + gradV(0,0)) - int(orderedPoints(0, 0))), 2) + pow((int(scanPt(0,1) + gradV(0,1)) - int(orderedPoints(0, 1))), 2);
            minDist2ZeroP = min(zero2RefDist, minDist2ZeroP);
        }
        if(objMask(int(scanPt(0,0) - gradV(0,0)), int(scanPt(0,1) - gradV(0,1))) == nc::uint8{0})
        {
            auto zero2RefDist = pow((int(scanPt(0,0) - gradV(0,0)) - int(orderedPoints(0, 0))), 2) + pow((int(scanPt(0,1) - gradV(0,1)) - int(orderedPoints(0, 1))), 2);
            minDist2ZeroP = min(zero2RefDist, minDist2ZeroP);
        }
        if(minDist2ZeroP < 10000.0)
            break;
        else
            {
                scanPt(0, 0) += normY;
                scanPt(0, 1) += normX;
            }
    }
    initiate0 = orderedPoints(0,0) + (-1*normY);
    initiate1 = orderedPoints(0,1) + (-1*normX);
    initiate0 = max(1+EPSILON, initiate0);
    initiate1 = max(1+EPSILON, initiate1);
    scanPt = nc::NdArray<double>{{min(initiate0, double(objMask.numRows() - 2 - EPSILON)),\
                                        min(initiate1, double(objMask.numCols() - 2 - EPSILON))}};
    double minDist2ZeroN = 10000.0;                      
    for(int k = 0; k<int(neckLengthTh /2 + 0.5); k++)
    {
        if(objMask(int(scanPt(0,0)), int(scanPt(0,1))) == nc::uint8{0})
        {
            auto zero2RefDist = pow((int(scanPt(0,0)) - int(orderedPoints(0, 0))), 2) + pow((int(scanPt(0,1)) - double(orderedPoints(0, 1))), 2);
            minDist2ZeroN = min(zero2RefDist, minDist2ZeroN);
        }
        if(objMask(int(scanPt(0,0) + gradV(0,0)), int(scanPt(0,1) + gradV(0,1))) == nc::uint8{0})
        {
            auto zero2RefDist = pow((int(scanPt(0,0) + gradV(0,0)) - int(orderedPoints(0, 0))), 2) + pow((int(scanPt(0,1) + gradV(0,1)) - int(orderedPoints(0, 1))), 2);
            minDist2ZeroN = min(zero2RefDist, minDist2ZeroN);
        }
        if(objMask(int(scanPt(0,0) - gradV(0,0)), int(scanPt(0,1) - gradV(0,1))) == nc::uint8{0})
        {
            auto zero2RefDist = pow((int(scanPt(0,0) - gradV(0,0)) - int(orderedPoints(0, 0))), 2) + pow((int(scanPt(0,1) - gradV(0,1)) - int(orderedPoints(0, 1))), 2);
            minDist2ZeroN = min(zero2RefDist, minDist2ZeroN);
        }
        if(minDist2ZeroN < 10000.0)
            break;
        else
            {
                scanPt(0, 0) += normY;
                scanPt(0, 1) += normX;
            }
    }
    if(minDist2ZeroP + minDist2ZeroN <= pow(neckLengthTh, 2))
        std::cout<<"minDist2ZeroP = "<<minDist2ZeroP<<", minDist2ZeroN = "<<minDist2ZeroN<<"\n";
    if(minDist2ZeroP + minDist2ZeroN <= double(neckLengthTh))
        truth = 1;
    else
        truth = 0;
    
    return scanNeigh;
}


nc::NdArray<nc::uint8> L2BreakNeck(nc::NdArray<nc::uint8> img, int virtualBrushSize)
{
    nc::NdArray<int> openEdges = {{0, 0}};
    nc::NdArray<int> indexes = {{2}};
    nc::NdArray<int> intersectEdges = {{0, 0}};
    float scanWFactor = 0.5;
    unsigned labelCount, oversamplingFactor;
    oversamplingFactor = 4;
    auto tempImage = img;
    auto skeletonImage = skeletonize(img);
    edgeParams(skeletonImage, img.numRows(), img.numCols(), openEdges, intersectEdges);
    if(intersectEdges.size() < 2)
    {
        return img;
    }
    auto componentPoints = getComponents(skeletonImage, intersectEdges, indexes, labelCount);
    for(unsigned i=1; i < labelCount; i++)
    {
        skeletonImage = nc::uint8{0};
        for(int j=indexes(0, i-1); j < indexes(0, i); j+=2)
            skeletonImage(componentPoints[j], componentPoints[j+1]) = nc::uint8{255};
        auto orderedPoints = L3SortSKCoordPts(skeletonImage);
        // orderedPoints.print();
        // To implement orderedPoints = L4CubicSpline2dUsingInHouse //
        auto dyDt = nc::gradient(orderedPoints(orderedPoints.rSlice(), 0), nc::Axis::ROW);
        dyDt = dyDt * double(-1);
        auto dxDt = nc::gradient(orderedPoints(orderedPoints.rSlice(), 1), nc::Axis::ROW);
        auto normVector = nc::stack({dyDt, dxDt}, nc::Axis::COL);
        // normVector.print();
        auto sizeV = nc::norm(normVector, nc::Axis::COL);
        auto normVectorY = nc::divide(normVector(normVector.rSlice(), 0), nc::transpose(sizeV));
        auto normVectorX = nc::divide(normVector(normVector.rSlice(), 1), nc::transpose(sizeV));
        for(unsigned norm = 0; norm < normVectorY.size(); norm++)
        {
            unsigned neckExists;
            auto scanNeigh = L3Check2BreakNecks(img, orderedPoints(norm, orderedPoints.cSlice()), normVectorY[norm], normVectorX[norm], virtualBrushSize, scanWFactor, neckExists);
            if(neckExists == 1)
            {
                for(unsigned neigh=0; neigh < scanNeigh.size(); neigh +=2)
                    tempImage(int(scanNeigh(0, neigh)), int(scanNeigh(0, neigh + 1))) = nc::uint8{0};
            }
        }
    }
    return tempImage;
}


void experiment()
{  
     
        // orderedPoints.print();
        // orderedPoints(0, orderedPoints.cSlice()).print();
    cv::Mat img;
    img = readImage("sample.png");
    nc::NdArray<nc::uint8> cmg = nc::NdArray<nc::uint8>(img.data, img.rows, img.cols);
    auto skeletonImage = L2BreakNeck(cmg, 5);
    cv::Mat tempImage = cv::Mat(skeletonImage.numRows(), skeletonImage.numCols(), CV_8U, skeletonImage.data());
    saveImage(tempImage, "ltp.png");   
}
// cv::String getDirection()

int main(int argc, char** argv)
{

    // auto maskImage = nc::NdArray<nc::uint8>{ {0,0,0,0,0,0},
    //                                         {0,255,255,255,255,0},
    //                                         {0,0,255,255,255,0},
    //                                         {0,0,0,0,0,0},
    //                                         {0,0,0,0,0,0},
    //                                         {0,255,255,0,0,0},
    //                                         {0,0,0,0,0,0} };
    // auto kernel = diskStructureKernel(1);
    // auto img = morphology(maskImage, kernel);
    
    experiment();
    // Parse Json
    // Json::Value root;
    // nc::NdArray<int> openEdges = {{0, 0}};
    // nc::NdArray<int> intersectEdges = {{0, 0}};
    // root = readJson("sample.json");
    // std::cout << root << std::endl;
    
    // Read Image
    // cv::Mat img, skelImg, labels;
    // unsigned labelCount;
    // img = readImage(root["ImagePath"].asString());
    // Convert Image to NumCpp array
    // nc::NdArray<nc::uint8> NcImage = nc::NdArray<nc::uint8>(img.data, img.rows, img.cols);

    // Retrieve image data
    // auto rows = img.rows;
    // auto columns =  img.cols;

    // Retrieve open edge coordinates and dangling edge coordinates
    // edgeParams(NcImage, rows, columns, openEdges, intersectEdges);
    // Retrieve components
    // getComponents(NcImage, intersectEdges, labels, labelCount);
    // std::cout<<labelCount<<"\n";
    // std::cout<<intersectEdges<<"\n";
    // showImage(img);

    // auto a = nc::random::randInt<int>({10, 10}, 0, 100);
    // std::cout << a;
    // Read the image file
    // Mat image = imread("skeletonHorse.png");

    // // Check for failure
    // if (image.empty()) 
    // {
    //     cout << "Could not open or find the image" << endl;
    //     cin.get(); //wait for any key press
    //     return -1;
    // }   

    // nc::NdArray<int> a = { {1, 2}, {3, 4}, {5, 6} };
    // std::cout<<nc::unique(a);

    return 0;
}