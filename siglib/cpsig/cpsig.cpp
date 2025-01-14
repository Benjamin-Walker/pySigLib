#include "cppch.h"
#include "cpsig.h"
#include <iostream>

#include "cpPath.h"
#include "cpTensorPoly.h"


double getPathElement(double* dataPtr, int dataLength, int dataDimension, int lengthIndex, int dimIndex) {
	Path<double> path(dataPtr, static_cast<uint64_t>(dataDimension), static_cast<uint64_t>(dataLength));
	return path[static_cast<uint64_t>(lengthIndex)][static_cast<uint64_t>(dimIndex)];
}
