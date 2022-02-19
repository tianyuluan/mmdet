<!--
 * @Author: luantianyu
 * @LastEditors: Luan Tianyu
 * @email: 1558747541@qq.com
 * @github: https://github.com/tianyuluan/
 * @Date: 2022-02-18 15:23:14
 * @LastEditTime: 2022-02-19 19:01:23
 * @motto: Still water run deep
 * @Description: Modify here please
 * @FilePath: /mmdet/README.md
-->
### mmdet
A dev mmdet version for BDD100k(https://arxiv.org/abs/1805.04687)

## A method to trans BDD100k-det label format to COCO label format

1: install bdd100k tools: https://github.com/bdd100k/bdd100k
    pip install bdd100k

2: install scalabel: https://github.com/scalabel/scalabel

3: BDD2COCO
    python bdd100k.label.to_coco det -i ... -o ....

