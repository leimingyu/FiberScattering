#ifndef UTIL_VECTOR_H
#define UTIL_VECTOR_H

struct float4
{
	float4() {}
	float4(float s) : x(s), y(s), z(s), w(s) {}
	float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	float x,y,z,w;

	inline float4 operator*(float s) const { 
		return float4(x*s, y*s, z*s, w*s);
	}

	inline float4 operator+(const float4 &a) const { 
		return float4(a.x + x, a.y + y, a.z + z, a.w + w); 
	}

	inline float4 operator-(const float4 &a) const {
		return float4(x - a.x, y - a.y, z - a.z, w - a.w); 
	}
};

inline float dot (const float4 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;	
}

	


#endif
