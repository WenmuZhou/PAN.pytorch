//
//  pse
//  Created by zhoujun on 11/9/19.
//  Copyright © 2019年 zhoujun. All rights reserved.
//
#include <queue>
#include <math.h>
#include <map>
#include <vector>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"

namespace py = pybind11;

namespace pan{
    py::array_t<uint8_t> pse(
    py::array_t<uint8_t, py::array::c_style> text,
    py::array_t<float, py::array::c_style> similarity_vectors,
    py::array_t<int32_t, py::array::c_style> label_map,
    float dis_threshold = 0.8)
    {
        auto pbuf_text = text.request();
        auto pbuf_similarity_vectors = similarity_vectors.request();
        auto pbuf_label_map = label_map.request();
        if (pbuf_label_map.ndim != 2 || pbuf_label_map.shape[0]==0 || pbuf_label_map.shape[1]==0)
            throw std::runtime_error("label map must have a shape of (h>0, w>0)");
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];
        if (pbuf_similarity_vectors.ndim != 3 || pbuf_similarity_vectors.shape[0]!=h || pbuf_similarity_vectors.shape[1]!=w || pbuf_similarity_vectors.shape[2]!=4 ||
            pbuf_text.shape[0]!=h || pbuf_text.shape[1]!=w)
            throw std::runtime_error("similarity_vectors must have a shape of (h,w,4) and text must have a shape of (h,w,4)");
        //初始化结果
        auto res = py::array_t<uint8_t>(pbuf_text.size);
        auto pbuf_res = res.request();
        // 获取 text similarity_vectors 和 label_map的指针
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_text = static_cast<uint8_t *>(pbuf_text.ptr);
        auto ptr_similarity_vectors = static_cast<float *>(pbuf_similarity_vectors.ptr);
        auto ptr_res = static_cast<uint8_t *>(pbuf_res.ptr);

        std::queue<std::tuple<int, int, int32_t>> q;
        // 计算各个kernel的similarity_vectors
        std::map<int,std::vector<float>> kernel_dict;
        std::map<int,std::vector<float>>::iterator iter;
        // 文本像素入队列
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            auto p_res = ptr_res + i*w;
            auto p_similarity_vectors = ptr_similarity_vectors + i*w*4;
            for(int j = 0, k = 0; j<w && k < w * 4; j++,k+=4)
            {
                int32_t label = p_label_map[j];
                if (label>0)
                {
                    std::vector<float> sv;
                    sv.push_back(p_similarity_vectors[k]);
                    sv.push_back(p_similarity_vectors[k+1]);
                    sv.push_back(p_similarity_vectors[k+2]);
                    sv.push_back(p_similarity_vectors[k+3]);
                    sv.push_back(1);
                    iter = kernel_dict.find(label);
                    if(iter != kernel_dict.end())
                    {
                        auto values = iter->second;
                        sv[0] += values[0];
                        sv[1] += values[1];
                        sv[2] += values[2];
                        sv[3] += values[3];
                        sv[4] += values[4];
                    }
                    kernel_dict[label] = sv;
                    q.push(std::make_tuple(i, j, label));
                }
                p_res[j] = label;
            }
        }

        for (auto& it : kernel_dict)
        {
            for (size_t i = 0;i<it.second.size() - 1;i++)
            {
                it.second[i] /= it.second[4];
            }
        }

        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        while(!q.empty()){
            //get each queue menber in q
            auto q_n = q.front();
            q.pop();
            int y = std::get<0>(q_n);
            int x = std::get<1>(q_n);
            int32_t l = std::get<2>(q_n);
            //store the edge pixel after one expansion
            auto kernel_cv = kernel_dict[l];
            for (int idx=0; idx<4; idx++)
            {
                int tmpy = y + dy[idx];
                int tmpx = x + dx[idx];
                auto p_res = ptr_res + tmpy*w;
                if (tmpy<0 || tmpy>=h || tmpx<0 || tmpx>=w)
                    continue;
                if (!ptr_text[tmpy*w+tmpx] || p_res[tmpx]>0)
                    continue;
                // 计算距离
                float dis = 0;
                auto p_similarity_vectors = ptr_similarity_vectors + tmpy * w*4;
                for(size_t i=0;i<kernel_cv.size()-1;i++)
                {
                    dis += pow(kernel_cv[i] - p_similarity_vectors[tmpx*4 + i],2);
                }
                dis = sqrt(dis);
                if(dis >= dis_threshold)
                    continue;
                q.push(std::make_tuple(tmpy, tmpx, l));
                p_res[tmpx]=l;
            }
        }
        return res;
    }
}

PYBIND11_MODULE(pse, m){
    m.def("pse_cpp", &pan::pse, " re-implementation pse algorithm(cpp)", py::arg("text"), py::arg("similarity_vectors"), py::arg("label_map"), py::arg("dis_threshold")=0.8);
}

