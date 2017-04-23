#include "fluids/common.hpp"
#include "fluids/gen_sphere.hpp"
#include <cmath>
#include <iostream>

#define FACE_LEFT   0
#define FACE_RIGHT  1
#define FACE_FRONT  2
#define FACE_BACK   3
#define FACE_TOP    4
#define FACE_BOTTOM 5
#define FACE_COUNT  6
#define VERTEX_SIZE 11

namespace Fluids {

    void generateSphere(int&    vertices,     // output number of vertices
                        int&    faces,        // output number of faces
                        GLuint& array_buffer, // output OpenGL array buffer
                        GLuint& index_buffer, // output OpenGL index buffer
                        float   radius,       // radius of the sphere
                        int     size)         // number of vertices
    {
        // --- Calculate Vertex Data -------------------------------------------------------

        vertices = 2 * size * size + 2 * size * (size - 2) + 2 * (size - 2) * (size - 2);
        if ( vertices > 256 ) {
            throw "Cannot allocate a sphere with that many vertices!";
        }
        faces = FACE_COUNT * 2 * size * size;

        int float_count = vertices * VERTEX_SIZE;
        int index_count = faces * 3;

        // --- Size Attributes -------------------------------------------------------------

        const int width = 1;
        const int length = size;
        const int height = size * size;

        const int vertex_offset = 0;
        const int normal_offset = vertices * 3;
        const int uv_offset = vertices * 6;
        const int color_offset = vertices * 8;

        // --- Generate Reference Buffers --------------------------------------------------

        byte** by_face = new byte*[FACE_COUNT];
        for (int i=0;i<FACE_COUNT;i++)
            by_face[i] = new byte[size * size];

        // --- Calculate Vertex Positions and fill reference buffers -----------------------

        float* vertexData = new float[float_count];
        int span=size-1;
        int x=0, y=0, z=0;

        for ( int i=0;i<vertices;i++ ) {
            if ( x == 0 )
                by_face[FACE_LEFT][y + z*size] = (byte)i;
            else if ( x == span )
                by_face[FACE_RIGHT][y + (size-z-1)*size] = (byte)i;

            if ( y == 0 )
                by_face[FACE_FRONT][x + (size-z-1)*size] = (byte)i;
            else if ( y == size-1 )
                by_face[FACE_BACK][x + z*size] = (byte)i;
            
            if ( z == 0 )
                by_face[FACE_TOP][x + y*size] = (byte)i;
            else if ( z == size-1 )
                by_face[FACE_BOTTOM][x + (size-y-1)*size] = (byte)i;

            float xp = -1.0f + x * (2.0f / span);
            float yp = -1.0f + y * (2.0f / (size-1));
            float zp = -1.0f + z * (2.0f / (size-1));

            float len = std::sqrt(xp*xp + yp*yp + zp*zp);

            xp /= len; yp /= len; zp /= len;

            vertexData[vertex_offset + 3*i +  0] = xp * radius;
            vertexData[vertex_offset + 3*i +  1] = yp * radius;
            vertexData[vertex_offset + 3*i +  2] = zp * radius;

            vertexData[normal_offset + 3*i +  0] = xp;
            vertexData[normal_offset + 3*i +  1] = yp;
            vertexData[normal_offset + 3*i +  2] = zp;

            vertexData[uv_offset + 2*i +  0] = 0;
            vertexData[uv_offset + 2*i +  1] = 0;

            vertexData[color_offset + 3*i + 0] = 1;
            vertexData[color_offset + 3*i + 1] = 1;
            vertexData[color_offset + 3*i + 2] = 1;

            if(++ x > span) {
                x = 0; if(++ y == size) {y = 0; if(++ z == size) z=0;}
                span = ((z==0||z==(size-1))||(y==0||y==(size-1)))?size-1:1;
            }
        }

        // --- Calculate Faces -------------------------------------------------------------

        byte* faceData = new byte[index_count];
        int index = 0;

        for ( int face=0;face<FACE_COUNT;face++ ) {
            for (int y=0;y<size-1;y++) {
                for (int x=0;x<size-1;x++) {
                    int anchor = x + y*size;

                    faceData[index+0] = by_face[face][anchor + 0];
                    faceData[index+1] = by_face[face][anchor + size];
                    faceData[index+2] = by_face[face][anchor + 1];

                    faceData[index+3] = by_face[face][anchor + size];
                    faceData[index+4] = by_face[face][anchor + size + 1];
                    faceData[index+5] = by_face[face][anchor + 1];

                    index += 6;
                }
            }
        }

        // --- Send buffer data to graphics library ----------------------------------------

        glGenBuffers(1, &array_buffer);
        glGenBuffers(1, &index_buffer);

        glBindBuffer(GL_ARRAY_BUFFER, array_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * float_count, vertexData, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_count, faceData, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // --- Clean up host buffers -------------------------------------------------------

        for (int i=0;i<FACE_COUNT;i++)
            delete[] by_face[i];
        delete[] by_face;

        delete[] faceData;
        delete[] vertexData;
    }

}