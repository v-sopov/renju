#ifdef PYTHON_C_EXTENTION
#include <Python.h>
#include <numpy/arrayobject.h>
#else
#include "python_include/Python.h"
#include "numpy_include/arrayobject.h"
#endif
#include <algorithm>
#include <ctime>
#include <vector>
#include <ctime>
#include <iostream>
#include <cstring>

bool equals(const float& a, const float& b) {
    return (a - b < 0.01f && a - b > -0.01f);
}

float get_elem(PyArrayObject *board, int row, int col) {
    float *ptr = (float *) PyArray_GETPTR2(board, row, col);
    return *ptr;
}

void set_elem(PyArrayObject *board, int row, int col, float value) {
    float *ptr = (float *) PyArray_GETPTR2(board, row, col);
    *ptr = value;
}

char get_neighbours(PyArrayObject *board, int row, int col, int drow, int dcol, float val) {
    char count = 0;
    int i;
    for (int iter = 0; iter < 2; ++iter) {
        i = 1;
        while (true) {
            if (row+i*drow >= 15 || col+i*dcol >= 15) {
                break;
            }
            float res = get_elem(board, row+i*drow, col+i*dcol);
            if (!equals(res, val)) {
                break;
            }
            ++count;
            ++i;
        }
        drow = -drow;
        dcol = -dcol;
    }
    return count;
}

int get_rollout_result_raw(PyArrayObject *board, const std::pair<int, int>& action) {
    if (action.first == -1) {
        return 2;
    }
    int row = action.first;
    int col = action.second;
    float *ptr;
    ptr = (float*)PyArray_GETPTR2(board, row, col);
    float val = *ptr;
    if (equals(val, 0.0f)) {
        return 2;
    }
    int j;
    int counts = 1;
    for (int i = row+1; i < 15; ++i) {
        ptr = (float*)PyArray_GETPTR2(board, i, col);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
    }
    for (int i = row-1; i >= 0; --i) {
        ptr = (float*)PyArray_GETPTR2(board, i, col);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
    }
    if (counts >= 5) {
        return (int)lround(val);
    }

    counts = 1;
    for (int i = col+1; i < 15; ++i) {
        ptr = (float*)PyArray_GETPTR2(board, row, i);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
    }
    for (int i = col-1; i >= 0; --i) {
        ptr = (float*)PyArray_GETPTR2(board, row, i);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
    }
    if (counts >= 5) {
        return (int)lround(val);
    }

    counts = 1;
    j = col+1;
    for (int i = row+1; i < 15; ++i) {
        if (j >= 15) {
            break;
        }
        ptr = (float*)PyArray_GETPTR2(board, i, j);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
        ++j;
    }
    j = col-1;
    for (int i = row-1; i >= 0; --i) {
        if (j < 0) {
            break;
        }
        ptr = (float*)PyArray_GETPTR2(board, i, j);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
        --j;
    }
    if (counts >= 5) {
        return (int)lround(val);
    }

    counts = 1;
    j = col-1;
    for (int i = row+1; i < 15; ++i) {
        if (j < 0) {
            break;
        }
        ptr = (float*)PyArray_GETPTR2(board, i, j);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
        --j;
    }
    j = col+1;
    for (int i = row-1; i >= 0; --i) {
        if (j >= 15) {
            break;
        }
        ptr = (float*)PyArray_GETPTR2(board, i, j);
        if (equals(*ptr, val)) {
            counts += 1;
        } else {
            break;
        }
        ++j;
    }
    if (counts >= 5) {
        return (int)lround(val);
    }

    float sum = 0.0f;
    for (int row = 0; row < 15; ++row) {
        for (int col = 0; col < 15; ++col) {
            float *ptr = (float*)PyArray_GETPTR2(board, row, col);
            float abs_elem = *ptr;
            abs_elem = (float)fabs(abs_elem);
            sum += abs_elem;
        }
    }
    if (equals(sum, 225.0f)) {
        return 0;
    } else {
        return 2;
    }
}

std::pair<std::pair<int, int>, std::pair<int, int> > get_positions(PyArrayObject *board, const std::pair<int, int>& point,
                                                                   std::vector<std::pair<int,int> >& threes,
                                                                   std::vector<std::pair<int,int> >& fours) {
    int row = point.first;
    int col = point.second;
    int length;
    float *ptr;
    ptr = (float *) PyArray_GETPTR2(board, row, col);
    float val = *ptr;
    int i1, i2;
    int j1, j2;
    //vertical
    for (i1 = row + 1; i1 < 15; ++i1) {
        ptr = (float *) PyArray_GETPTR2(board, i1, col);
        if (!equals(*ptr, val)) {
            break;
        }
    }
    for (i2 = row - 1; i2 >= 0; --i2) {
        ptr = (float *) PyArray_GETPTR2(board, i2, col);
        if (!equals(*ptr, val)) {
            break;
        }
    }
    length = i1 - i2 - 1;
    if (length == 3 || length == 4) {
        i1 = std::min(14, i1);
        i2 = std::max(0, i2);
        ptr = (float*)PyArray_GETPTR2(board, i1, col);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(i1, col);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
        ptr = (float*)PyArray_GETPTR2(board, i2, col);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(i2, col);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
    }
    //horizontal
    for (i1 = col + 1; i1 < 15; ++i1) {
        ptr = (float *) PyArray_GETPTR2(board, row, i1);
        if (!equals(*ptr, val)) {
            break;
        }
    }
    for (i2 = col - 1; i2 >= 0; --i2) {
        ptr = (float *) PyArray_GETPTR2(board, row, i2);
        if (!equals(*ptr, val)) {
            break;
        }
    }
    length = i1 - i2 - 1;
    if (length == 3 || length == 4) {
        i1 = std::min(14, i1);
        i2 = std::max(0, i2);
        ptr = (float*)PyArray_GETPTR2(board, row, i1);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(row, i1);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
        ptr = (float*)PyArray_GETPTR2(board, row, i2);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(row, i2);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
    }
    //to-right-bottom
    j1 = col + 1;
    for (i1 = row + 1; i1 < 15; ++i1) {
        if (j1 >= 15) {
            break;
        }
        ptr = (float *) PyArray_GETPTR2(board, i1, j1);
        if (!equals(*ptr, val)) {
            break;
        }
        ++j1;
    }
    j2 = col - 1;
    for (i2 = row - 1; i2 >= 0; --i2) {
        if (j2 < 0) {
            break;
        }
        ptr = (float *) PyArray_GETPTR2(board, i2, j2);
        if (!equals(*ptr, val)) {
            break;
        }
        --j2;
    }
    length = i1 - i2 - 1;
    if (length == 3 || length == 4) {
        if (i1 == 15 || j1 == 15) {
            --i1;
            --j1;
        }
        if (i2 < 0 || j2 < 0) {
            ++i2;
            ++j2;
        }
        ptr = (float*)PyArray_GETPTR2(board, i1, j1);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(i1, j1);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
        ptr = (float*)PyArray_GETPTR2(board, i2, j2);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(i2, j2);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
    }
    //to-left-bottom
    j1 = col - 1;
    for (i1 = row + 1; i1 < 15; ++i1) {
        if (j1 < 0) {
            break;
        }
        ptr = (float *) PyArray_GETPTR2(board, i1, j1);
        if (!equals(*ptr, val)) {
            break;
        }
        --j1;
    }
    j2 = col + 1;
    for (i2 = row - 1; i2 >= 0; --i2) {
        if (j2 >= 15) {
            break;
        }
        ptr = (float *) PyArray_GETPTR2(board, i2, j2);
        if (!equals(*ptr, val)) {
            break;
        }
        ++j2;
    }
    length = i1 - i2 - 1;
    if (length == 3 || length == 4) {
        if (i1 == 15 || j1 < 0) {
            --i1;
            ++j1;
        }
        if (i2 < 0 || j2 == 15) {
            ++i2;
            --j2;
        }
        ptr = (float*)PyArray_GETPTR2(board, i1, j1);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(i1, j1);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
        ptr = (float*)PyArray_GETPTR2(board, i2, j2);
        if (equals(0.0f, *ptr)) {
            auto res = std::make_pair(i2, j2);
            if (length == 3) {
                threes.push_back(res);
            } else {
                fours.push_back(res);
            }
        }
    }
}

extern "C" {

PyObject* mcts_utils_random_near(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    PyArrayObject *arr = (PyArrayObject*)obj;
    std::vector<int> near;
    std::srand(unsigned(std::time(0)));
    for (int row = 0; row < 15; ++row) {
        for (int col = 0; col < 15; ++col) {
            float *ptr = (float*)PyArray_GETPTR2(arr, row, col);
            if (*ptr > 0.01f || *ptr < -0.01f) {
                for (int i = std::max(0, row-1); i < std::min(15, row+2); ++i) {
                    for (int j = std::max(0, col-1); j < std::min(15, col+2); ++j) {
                        ptr = (float*)PyArray_GETPTR2(arr, i, j);
                        if (*ptr < 0.01f && *ptr > -0.01f) {
                            near.push_back(i*15 + j);
                        }
                    }
                }
            }
        }
    }
    int action;
    if (near.empty()) {
        action = 7*15 + 7;
    } else {
        action = near[std::rand() % near.size()];
    }
    return Py_BuildValue("(i, i)", action / 15, action % 15);
}


PyObject* mcts_utils_clever_near(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj))
        return NULL;
    PyArrayObject *arr = (PyArrayObject*)obj;
    std::vector<std::pair<int,int> > threes;
    std::vector<std::pair<int,int> > fours;
    std::vector<int> near;
    std::srand(unsigned(std::time(0)));
    for (int row = 0; row < 15; ++row) {
        if (!fours.empty() || !threes.empty()) {
            break;
        }
        for (int col = 0; col < 15; ++col) {
            float *ptr = (float *) PyArray_GETPTR2(arr, row, col);
            if (!equals(*ptr, 0.0f)) {
                get_positions(arr, {row, col}, threes, fours);
                if (!fours.empty() || !threes.empty()) {
                    break;
                }
                for (int i = std::max(0, row - 1); i < std::min(15, row + 2); ++i) {
                    for (int j = std::max(0, col - 1); j < std::min(15, col + 2); ++j) {
                        ptr = (float *) PyArray_GETPTR2(arr, i, j);
                        if (*ptr < 0.01f && *ptr > -0.01f) {
                            near.push_back(i * 15 + j);
                        }
                    }
                }
            }
        }
    }
    if (!fours.empty()) {
        auto res = fours[std::rand() % fours.size()];
        return Py_BuildValue("(i, i)", res.first, res.second);
    }
    if (!threes.empty()) {
        auto res = threes[std::rand() % threes.size()];
        return Py_BuildValue("(i, i)", res.first, res.second);
    }
    int action;
    if (near.empty()) {
        action = 7*15 + 7;
    } else {
        action = near[std::rand() % near.size()];
    }
    return Py_BuildValue("(i, i)", action / 15, action % 15);
}

PyObject* mcts_utils_get_rollout_result(PyObject *self, PyObject *args) {
    PyObject *board, *tuple;
    if (!PyArg_ParseTuple(args, "OO", &board, &tuple))
        return NULL;
    if (tuple == Py_None) {
        Py_RETURN_NONE;
    }
    PyObject *first = PyTuple_GetItem(tuple, 0);
    PyObject *second = PyTuple_GetItem(tuple, 1);
    std::pair<int, int> action = {PyLong_AsLong(first), PyLong_AsLong(second)};

    int res = get_rollout_result_raw((PyArrayObject*)board, action);
    if (res != 2) {
        return Py_BuildValue("i", res);
    } else {
        Py_RETURN_NONE;
    }
}


PyObject* mcts_utils_full_near(PyObject *self, PyObject *args) {
    PyObject *obj;
    bool is_black;
    if (!PyArg_ParseTuple(args, "Ob", &obj, &is_black))
        return NULL;
    PyArrayObject *board = (PyArrayObject*)obj;
    char neighbours[15][15][4][2];
    char total_neighbours[15][15][2];
    char own_four[2] = {-1, -1}, enemy_four[2] = {-1, -1}, three[2] = {-1, -1};
    std::vector<std::pair<char,char>> actions[2];
    std::memset(neighbours, 0, 15*15*4*2);
    std::memset(total_neighbours, 0, 15*15*2);
    for (char row = 0; row < 15; ++row) {
        for (char col = 0; col < 15; ++col) {
            if (!equals(get_elem(board, row, col), 0.0f)) {
                continue;
            }
            neighbours[row][col][0][0] += get_neighbours(board, row, col, 0, 1, -1.0f);
            neighbours[row][col][0][1] += get_neighbours(board, row, col, 0, 1, 1.0f);
            neighbours[row][col][1][0] += get_neighbours(board, row, col, 1, 0, -1.0f);
            neighbours[row][col][1][1] += get_neighbours(board, row, col, 1, 0, 1.0f);
            neighbours[row][col][2][0] += get_neighbours(board, row, col, -1, 1, -1.0f);
            neighbours[row][col][2][1] += get_neighbours(board, row, col, -1, 1, 1.0f);
            neighbours[row][col][3][0] += get_neighbours(board, row, col, 1, 1, -1.0f);
            neighbours[row][col][3][1] += get_neighbours(board, row, col, 1, 1, 1.0f);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    total_neighbours[row][col][j] = std::max(total_neighbours[row][col][j], neighbours[row][col][i][j]);
                }
            }
            if (own_four[0] == -1 && total_neighbours[row][col][is_black ? 0 : 1] >= 4) {
                own_four[0] = row;
                own_four[1] = col;
                break;
            } else if ((enemy_four[0] == -1 || std::rand() % 2 == 0) && total_neighbours[row][col][is_black ? 1 : 0] >= 4) {
                enemy_four[0] = row;
                enemy_four[1] = col;
            } else if ((three[0] == -1 || std::rand() % 2 == 0) && std::max(total_neighbours[row][col][0], total_neighbours[row][col][1]) >= 3) {
                three[0] = row;
                three[1] = col;
            }
            if (std::max(total_neighbours[row][col][0], total_neighbours[row][col][1]) == 2) {
                actions[0].emplace_back(std::make_pair(row, col));
            } else if (std::max(total_neighbours[row][col][0], total_neighbours[row][col][1]) == 1) {
                actions[1].emplace_back(std::make_pair(row, col));
            }
        }
        if (own_four[0] != -1) {
            break;
        }
    }
    int row = -1, col = -1;
    if (own_four[0] != -1) {
        row = own_four[0];
        col = own_four[1];
    } else if (enemy_four[0] != -1) {
        row = enemy_four[0];
        col = enemy_four[1];
    } else if (three[0] != -1) {
        row = three[0];
        col = three[1];
    }
    if (row == -1) {
        if (!actions[0].empty()) {
            auto p = actions[0][std::rand() % actions[0].size()];
            row = p.first;
            col = p.second;
        } else if (!actions[1].empty()) {
            auto p = actions[1][std::rand() % actions[1].size()];
            row = p.first;
            col = p.second;
        } else {
            row = 7;
            col = 7;
        }
    }
    return Py_BuildValue("(i, i)", row, col);
}

PyObject* mcts_utils_find_fours(PyObject *self, PyObject *args) {
    PyObject *obj;
    bool is_black;
    if (!PyArg_ParseTuple(args, "Ob", &obj, &is_black))
        return NULL;
    PyArrayObject *board = (PyArrayObject*)obj;
    float player_val = is_black ? -1.0f : 1.0f;
    char neighbours[15][15][4][2];
    char total_neighbours[15][15][2];
    char own_four[2] = {-1, -1}, enemy_four[2] = {-1, -1};
    std::memset(neighbours, 0, 15*15*4*2);
    std::memset(total_neighbours, 0, 15*15*2);
    for (char row = 0; row < 15; ++row) {
        for (char col = 0; col < 15; ++col) {
            if (!equals(get_elem(board, row, col), 0.0f)) {
                continue;
            }
            neighbours[row][col][0][0] += get_neighbours(board, row, col, 0, 1, -1.0f);
            neighbours[row][col][0][1] += get_neighbours(board, row, col, 0, 1, 1.0f);
            neighbours[row][col][1][0] += get_neighbours(board, row, col, 1, 0, -1.0f);
            neighbours[row][col][1][1] += get_neighbours(board, row, col, 1, 0, 1.0f);
            neighbours[row][col][2][0] += get_neighbours(board, row, col, -1, 1, -1.0f);
            neighbours[row][col][2][1] += get_neighbours(board, row, col, -1, 1, 1.0f);
            neighbours[row][col][3][0] += get_neighbours(board, row, col, 1, 1, -1.0f);
            neighbours[row][col][3][1] += get_neighbours(board, row, col, 1, 1, 1.0f);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 2; ++j) {
                    total_neighbours[row][col][j] = std::max(total_neighbours[row][col][j], neighbours[row][col][i][j]);
                }
            }
            if (own_four[0] == -1 && total_neighbours[row][col][is_black ? 0 : 1] >= 4) {
                own_four[0] = row;
                own_four[1] = col;
                break;
            } else if ((enemy_four[0] == -1 || std::rand() % 2 == 0) && total_neighbours[row][col][is_black ? 1 : 0] >= 4) {
                enemy_four[0] = row;
                enemy_four[1] = col;
            }
        }
        if (own_four[0] != -1) {
            break;
        }
    }
    int row = -1, col = -1;
    if (own_four[0] != -1) {
        row = own_four[0];
        col = own_four[1];
    } else if (enemy_four[0] != -1) {
        row = enemy_four[0];
        col = enemy_four[1];
    }
    if (row != -1) {
        return Py_BuildValue("(i, i)", row, col);
    } else {
        Py_RETURN_NONE;
    }
}


PyMethodDef random_near_funcs[] = {
        {"random_near", (PyCFunction) mcts_utils_random_near, METH_VARARGS, NULL},
        {"clever_near", (PyCFunction) mcts_utils_clever_near, METH_VARARGS, NULL},
        {"full_near", (PyCFunction) mcts_utils_full_near, METH_VARARGS, NULL},
        {"find_fours", (PyCFunction) mcts_utils_find_fours, METH_VARARGS, NULL},
        {"get_rollout_result", (PyCFunction) mcts_utils_get_rollout_result, METH_VARARGS, NULL},
        {NULL, NULL,                               0, NULL}
};

static struct PyModuleDef mcts_utils_module = {
        PyModuleDef_HEAD_INIT,
        "mcts_utils",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        random_near_funcs
};

PyMODINIT_FUNC PyInit_mcts_utils(void) {
    std::srand((unsigned)std::time(NULL));
    import_array();
    return PyModule_Create(&mcts_utils_module);
}
}