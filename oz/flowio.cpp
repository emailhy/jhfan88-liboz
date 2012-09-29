// flow_io.cpp
//
// read and write our simple .flo flow file format

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//
#include <oz/flowio.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>


// the "official" threshold - if the absolute value of either 
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e9

// value to use to represent unknown flow
#define UNKNOWN_FLOW 1e10

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file


#ifdef _MSC_VER
namespace std { inline int isnan(float x) { return _isnanf(x); } }
#endif

// return whether flow vector is unknown
static bool unknown_flow(float u, float v) {
    return (fabs(u) >  UNKNOWN_FLOW_THRESH) 
    || (fabs(v) >  UNKNOWN_FLOW_THRESH)
    || std::isnan(u) || std::isnan(v);
}

static bool unknown_flow(float *f) {
    return unknown_flow(f[0], f[1]);
}


// read a flow file into 2-band image
oz::cpu_image oz::read_flow( const std::string& path ) {
    FILE *stream = 0;
    try {
        stream = fopen(path.c_str(), "rb");
        if (!stream) OZ_X() << "Opening file '" << path << "' failed!";
        
        int width, height;
        float tag;

        if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
            (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
            (int)fread(&height, sizeof(int),   1, stream) != 1)
            OZ_X() << "Invalid file!";

        if (tag != TAG_FLOAT) // simple test for correct endian-ness
            OZ_X() << "Wrong tag (possibly due to big-endian machine?)";

        // another sanity check to see that integers were read correctly (99999 should do the trick...)
        if (width < 1 || width > 99999)
            OZ_X() << "Illegal width!";

        if (height < 1 || height > 99999)
            OZ_X() << "Illegal height!";

        cpu_image sh(width, height, FMT_FLOAT2);

        //printf("reading %d x %d x 2 = %d floats\n", width, height, width*height*2);
        int n = 2 * width;
        for (int y = 0; y < height; y++) {
            float* ptr = sh.scan_line<float>(y);
            if ((int)fread(ptr, sizeof(float), n, stream) != n)
                OZ_X() << "Read error!";
        }

        if (fgetc(stream) != EOF)
            OZ_X() << "Expecting EOF!";

        fclose(stream);
        return sh;
    }
    catch (std::exception&) {
        if (stream) fclose(stream);
        throw;
    }
}


#if 0

// write a 2-band image into flow file 
void WriteFlowFile(CFloatImage img, const char* filename)
{
    if (filename == NULL)
    throw CError("WriteFlowFile: empty filename");

    char *dot = strrchr(filename, '.');
    if (dot == NULL)
    throw CError("WriteFlowFile: extension required in filename '%s'", filename);

    if (strcmp(dot, ".flo") != 0)
    throw CError("WriteFlowFile: filename '%s' should have extension '.flo'", filename);

    CShape sh = img.Shape();
    int width = sh.width, height = sh.height, nBands = sh.nBands;

    if (nBands != 2)
    throw CError("WriteFlowFile(%s): image must have 2 bands", filename);

    FILE *stream = fopen(filename, "wb");
    if (stream == 0)
        throw CError("WriteFlowFile: could not open %s", filename);

    // write the header
    fprintf(stream, TAG_STRING);
    if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
    (int)fwrite(&height, sizeof(int),   1, stream) != 1)
    throw CError("WriteFlowFile(%s): problem writing header", filename);

    // write the rows
    int n = nBands * width;
    for (int y = 0; y < height; y++) {
    float* ptr = &img.Pixel(0, y, 0);
    if ((int)fwrite(ptr, sizeof(float), n, stream) != n)
        throw CError("WriteFlowFile(%s): problem writing data", filename); 
   }

    fclose(stream);
}


#endif
