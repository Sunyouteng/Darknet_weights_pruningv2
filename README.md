main explaination:
本项目的主要目的是去除darknet框架中的动态内存分配和函数指针，使其可以用于综合出FPGA电路实现。
实现过程中，根据自己的理解，对框架进行了一些解释和说明。
最终的用于前向传播的net.layer的每一层构建应该分为两部分：
1	在 parse_network_cfg_custom(cfgfile, 1); 这里构建net的整体结构
2	load_weights 中加载每层layer的权值
3	