#include <assert.h>
#include <iostream>
#include <string>

void readMBR(char *file,
             const int size,
             const float r,
             float *&xmin,
             float *&ymin,
             float *&xmax,
             float *&ymax,
             int *&id)

{
  FILE *fp = fopen(file, "r");
  if (NULL == fp) {
    printf("open %s failed\n", file);
    exit(1);
  }
  xmin = new float[size];
  ymin = new float[size];
  xmax = new float[size];
  ymax = new float[size];
  id   = new int[size];
  int pid, nume;

  float x1, y1, x2, y2;
  for (int i = 0; i < size; i++) {
    int ret = fscanf(fp, "%f,%f,%f,%f,%d,%d,%d\n", &x1, &y1, &x2, &y2, &id[i], &pid, &nume);
    if (ret != 7) {
      throw std::logic_error("expected ret to be 7. received " + std::to_string(ret));
    }

    xmin[i] = std::min(x1, x2) - r;
    ymin[i] = std::min(y1, y2) - r;
    xmax[i] = std::max(x1, x2) + r;
    ymax[i] = std::max(y1, y2) + r;
  }
  fclose(fp);
}
