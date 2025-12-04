import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.channelshuffle_0 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_1 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_2 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_3 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_4 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_5 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_6 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_7 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_8 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_9 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_10 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_11 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_12 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_13 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_14 = nn.ChannelShuffle(groups=2)
        self.channelshuffle_15 = nn.ChannelShuffle(groups=2)

        archive = zipfile.ZipFile('nanodet_320_fp16.pnnx.bin', 'r')
        self.onnx__Conv_1252_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1252.data', (24,3,3,3,), 'float16')
        self.onnx__Conv_1253_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1253.data', (24,), 'float16')
        self.onnx__Conv_1255_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1255.data', (24,1,3,3,), 'float16')
        self.onnx__Conv_1256_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1256.data', (24,), 'float16')
        self.onnx__Conv_1258_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1258.data', (58,24,1,1,), 'float16')
        self.onnx__Conv_1259_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1259.data', (58,), 'float16')
        self.onnx__Conv_1261_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1261.data', (58,24,1,1,), 'float16')
        self.onnx__Conv_1262_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1262.data', (58,), 'float16')
        self.onnx__Conv_1264_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1264.data', (58,1,3,3,), 'float16')
        self.onnx__Conv_1265_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1265.data', (58,), 'float16')
        self.onnx__Conv_1267_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1267.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1268_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1268.data', (58,), 'float16')
        self.onnx__Conv_1270_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1270.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1271_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1271.data', (58,), 'float16')
        self.onnx__Conv_1273_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1273.data', (58,1,3,3,), 'float16')
        self.onnx__Conv_1274_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1274.data', (58,), 'float16')
        self.onnx__Conv_1276_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1276.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1277_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1277.data', (58,), 'float16')
        self.onnx__Conv_1279_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1279.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1280_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1280.data', (58,), 'float16')
        self.onnx__Conv_1282_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1282.data', (58,1,3,3,), 'float16')
        self.onnx__Conv_1283_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1283.data', (58,), 'float16')
        self.onnx__Conv_1285_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1285.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1286_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1286.data', (58,), 'float16')
        self.onnx__Conv_1288_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1288.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1289_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1289.data', (58,), 'float16')
        self.onnx__Conv_1291_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1291.data', (58,1,3,3,), 'float16')
        self.onnx__Conv_1292_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1292.data', (58,), 'float16')
        self.onnx__Conv_1294_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1294.data', (58,58,1,1,), 'float16')
        self.onnx__Conv_1295_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1295.data', (58,), 'float16')
        self.onnx__Conv_1297_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1297.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1298_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1298.data', (116,), 'float16')
        self.onnx__Conv_1300_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1300.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1301_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1301.data', (116,), 'float16')
        self.onnx__Conv_1303_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1303.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1304_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1304.data', (116,), 'float16')
        self.onnx__Conv_1306_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1306.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1307_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1307.data', (116,), 'float16')
        self.onnx__Conv_1309_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1309.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1310_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1310.data', (116,), 'float16')
        self.onnx__Conv_1312_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1312.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1313_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1313.data', (116,), 'float16')
        self.onnx__Conv_1315_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1315.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1316_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1316.data', (116,), 'float16')
        self.onnx__Conv_1318_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1318.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1319_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1319.data', (116,), 'float16')
        self.onnx__Conv_1321_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1321.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1322_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1322.data', (116,), 'float16')
        self.onnx__Conv_1324_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1324.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1325_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1325.data', (116,), 'float16')
        self.onnx__Conv_1327_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1327.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1328_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1328.data', (116,), 'float16')
        self.onnx__Conv_1330_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1330.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1331_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1331.data', (116,), 'float16')
        self.onnx__Conv_1333_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1333.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1334_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1334.data', (116,), 'float16')
        self.onnx__Conv_1336_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1336.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1337_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1337.data', (116,), 'float16')
        self.onnx__Conv_1339_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1339.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1340_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1340.data', (116,), 'float16')
        self.onnx__Conv_1342_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1342.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1343_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1343.data', (116,), 'float16')
        self.onnx__Conv_1345_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1345.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1346_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1346.data', (116,), 'float16')
        self.onnx__Conv_1348_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1348.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1349_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1349.data', (116,), 'float16')
        self.onnx__Conv_1351_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1351.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1352_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1352.data', (116,), 'float16')
        self.onnx__Conv_1354_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1354.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1355_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1355.data', (116,), 'float16')
        self.onnx__Conv_1357_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1357.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1358_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1358.data', (116,), 'float16')
        self.onnx__Conv_1360_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1360.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1361_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1361.data', (116,), 'float16')
        self.onnx__Conv_1363_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1363.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1364_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1364.data', (116,), 'float16')
        self.onnx__Conv_1366_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1366.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1367_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1367.data', (116,), 'float16')
        self.onnx__Conv_1369_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1369.data', (116,1,3,3,), 'float16')
        self.onnx__Conv_1370_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1370.data', (116,), 'float16')
        self.onnx__Conv_1372_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1372.data', (116,116,1,1,), 'float16')
        self.onnx__Conv_1373_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1373.data', (116,), 'float16')
        self.onnx__Conv_1375_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1375.data', (232,1,3,3,), 'float16')
        self.onnx__Conv_1376_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1376.data', (232,), 'float16')
        self.onnx__Conv_1378_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1378.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1379_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1379.data', (232,), 'float16')
        self.onnx__Conv_1381_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1381.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1382_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1382.data', (232,), 'float16')
        self.onnx__Conv_1384_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1384.data', (232,1,3,3,), 'float16')
        self.onnx__Conv_1385_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1385.data', (232,), 'float16')
        self.onnx__Conv_1387_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1387.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1388_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1388.data', (232,), 'float16')
        self.onnx__Conv_1390_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1390.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1391_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1391.data', (232,), 'float16')
        self.onnx__Conv_1393_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1393.data', (232,1,3,3,), 'float16')
        self.onnx__Conv_1394_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1394.data', (232,), 'float16')
        self.onnx__Conv_1396_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1396.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1397_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1397.data', (232,), 'float16')
        self.onnx__Conv_1399_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1399.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1400_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1400.data', (232,), 'float16')
        self.onnx__Conv_1402_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1402.data', (232,1,3,3,), 'float16')
        self.onnx__Conv_1403_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1403.data', (232,), 'float16')
        self.onnx__Conv_1405_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1405.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1406_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1406.data', (232,), 'float16')
        self.onnx__Conv_1408_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1408.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1409_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1409.data', (232,), 'float16')
        self.onnx__Conv_1411_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1411.data', (232,1,3,3,), 'float16')
        self.onnx__Conv_1412_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1412.data', (232,), 'float16')
        self.onnx__Conv_1414_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1414.data', (232,232,1,1,), 'float16')
        self.onnx__Conv_1415_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1415.data', (232,), 'float16')
        self.lateral_convs_2_weight_data = self.load_pnnx_bin_as_parameter(archive, 'lateral_convs.2.weight.data', (96,464,1,1,), 'float16')
        self.lateral_convs_2_bias_data = self.load_pnnx_bin_as_parameter(archive, 'lateral_convs.2.bias.data', (96,), 'float16')
        self.lateral_convs_1_weight_data = self.load_pnnx_bin_as_parameter(archive, 'lateral_convs.1.weight.data', (96,232,1,1,), 'float16')
        self.lateral_convs_1_bias_data = self.load_pnnx_bin_as_parameter(archive, 'lateral_convs.1.bias.data', (96,), 'float16')
        self.lateral_convs_0_weight_data = self.load_pnnx_bin_as_parameter(archive, 'lateral_convs.0.weight.data', (96,116,1,1,), 'float16')
        self.lateral_convs_0_bias_data = self.load_pnnx_bin_as_parameter(archive, 'lateral_convs.0.bias.data', (96,), 'float16')
        self.onnx__Conv_1417_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1417.data', (96,96,3,3,), 'float16')
        self.onnx__Conv_1418_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1418.data', (96,), 'float16')
        self.onnx__Conv_1420_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1420.data', (96,96,3,3,), 'float16')
        self.onnx__Conv_1421_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1421.data', (96,), 'float16')
        self.onnx__Conv_1423_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1423.data', (96,96,3,3,), 'float16')
        self.onnx__Conv_1424_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1424.data', (96,), 'float16')
        self.onnx__Conv_1426_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1426.data', (96,96,3,3,), 'float16')
        self.onnx__Conv_1427_data = self.load_pnnx_bin_as_parameter(archive, 'onnx::Conv_1427.data', (96,), 'float16')
        self.gfl_cls_weight_data = self.load_pnnx_bin_as_parameter(archive, 'gfl_cls.weight.data', (1,96,3,3,), 'float16')
        self.gfl_cls_bias_data = self.load_pnnx_bin_as_parameter(archive, 'gfl_cls.bias.data', (1,), 'float16')
        self.gfl_reg_weight_data = self.load_pnnx_bin_as_parameter(archive, 'gfl_reg.weight.data', (4,96,3,3,), 'float16')
        self.gfl_reg_bias_data = self.load_pnnx_bin_as_parameter(archive, 'gfl_reg.bias.data', (4,), 'float16')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.onnx__Conv_1252_data
        v_2 = self.onnx__Conv_1253_data
        v_3 = Conv(v_0, v_1, v_2, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_4 = F.relu(v_3)
        v_5 = F.max_pool2d(v_4, ceil_mode=False, dilation=(1,1), kernel_size=(3,3), padding=(1,1), return_indices=False, stride=(2,2))
        v_6 = self.onnx__Conv_1255_data
        v_7 = self.onnx__Conv_1256_data
        v_8 = Conv(v_5, v_6, v_7, dilations=(1,1), group=24, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_9 = self.onnx__Conv_1258_data
        v_10 = self.onnx__Conv_1259_data
        v_11 = Conv(v_8, v_9, v_10, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_12 = F.relu(v_11)
        v_13 = self.onnx__Conv_1261_data
        v_14 = self.onnx__Conv_1262_data
        v_15 = Conv(v_5, v_13, v_14, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_16 = F.relu(v_15)
        v_17 = self.onnx__Conv_1264_data
        v_18 = self.onnx__Conv_1265_data
        v_19 = Conv(v_16, v_17, v_18, dilations=(1,1), group=58, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_20 = self.onnx__Conv_1267_data
        v_21 = self.onnx__Conv_1268_data
        v_22 = Conv(v_19, v_20, v_21, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_23 = F.relu(v_22)
        v_24 = torch.cat((v_12, v_23), dim=1)
        v_25 = self.channelshuffle_0(v_24)
        v_26, v_27 = torch.tensor_split(v_25, dim=1, indices=(58,))
        v_28 = self.onnx__Conv_1270_data
        v_29 = self.onnx__Conv_1271_data
        v_30 = Conv(v_27, v_28, v_29, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_31 = F.relu(v_30)
        v_32 = self.onnx__Conv_1273_data
        v_33 = self.onnx__Conv_1274_data
        v_34 = Conv(v_31, v_32, v_33, dilations=(1,1), group=58, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_35 = self.onnx__Conv_1276_data
        v_36 = self.onnx__Conv_1277_data
        v_37 = Conv(v_34, v_35, v_36, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_38 = F.relu(v_37)
        v_39 = torch.cat((v_26, v_38), dim=1)
        v_40 = self.channelshuffle_1(v_39)
        v_41, v_42 = torch.tensor_split(v_40, dim=1, indices=(58,))
        v_43 = self.onnx__Conv_1279_data
        v_44 = self.onnx__Conv_1280_data
        v_45 = Conv(v_42, v_43, v_44, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_46 = F.relu(v_45)
        v_47 = self.onnx__Conv_1282_data
        v_48 = self.onnx__Conv_1283_data
        v_49 = Conv(v_46, v_47, v_48, dilations=(1,1), group=58, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_50 = self.onnx__Conv_1285_data
        v_51 = self.onnx__Conv_1286_data
        v_52 = Conv(v_49, v_50, v_51, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_53 = F.relu(v_52)
        v_54 = torch.cat((v_41, v_53), dim=1)
        v_55 = self.channelshuffle_2(v_54)
        v_56, v_57 = torch.tensor_split(v_55, dim=1, indices=(58,))
        v_58 = self.onnx__Conv_1288_data
        v_59 = self.onnx__Conv_1289_data
        v_60 = Conv(v_57, v_58, v_59, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_61 = F.relu(v_60)
        v_62 = self.onnx__Conv_1291_data
        v_63 = self.onnx__Conv_1292_data
        v_64 = Conv(v_61, v_62, v_63, dilations=(1,1), group=58, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_65 = self.onnx__Conv_1294_data
        v_66 = self.onnx__Conv_1295_data
        v_67 = Conv(v_64, v_65, v_66, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_68 = F.relu(v_67)
        v_69 = torch.cat((v_56, v_68), dim=1)
        v_70 = self.channelshuffle_3(v_69)
        v_71 = self.onnx__Conv_1297_data
        v_72 = self.onnx__Conv_1298_data
        v_73 = Conv(v_70, v_71, v_72, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_74 = self.onnx__Conv_1300_data
        v_75 = self.onnx__Conv_1301_data
        v_76 = Conv(v_73, v_74, v_75, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_77 = F.relu(v_76)
        v_78 = self.onnx__Conv_1303_data
        v_79 = self.onnx__Conv_1304_data
        v_80 = Conv(v_70, v_78, v_79, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_81 = F.relu(v_80)
        v_82 = self.onnx__Conv_1306_data
        v_83 = self.onnx__Conv_1307_data
        v_84 = Conv(v_81, v_82, v_83, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_85 = self.onnx__Conv_1309_data
        v_86 = self.onnx__Conv_1310_data
        v_87 = Conv(v_84, v_85, v_86, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_88 = F.relu(v_87)
        v_89 = torch.cat((v_77, v_88), dim=1)
        v_90 = self.channelshuffle_4(v_89)
        v_91, v_92 = torch.tensor_split(v_90, dim=1, indices=(116,))
        v_93 = self.onnx__Conv_1312_data
        v_94 = self.onnx__Conv_1313_data
        v_95 = Conv(v_92, v_93, v_94, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_96 = F.relu(v_95)
        v_97 = self.onnx__Conv_1315_data
        v_98 = self.onnx__Conv_1316_data
        v_99 = Conv(v_96, v_97, v_98, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_100 = self.onnx__Conv_1318_data
        v_101 = self.onnx__Conv_1319_data
        v_102 = Conv(v_99, v_100, v_101, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_103 = F.relu(v_102)
        v_104 = torch.cat((v_91, v_103), dim=1)
        v_105 = self.channelshuffle_5(v_104)
        v_106, v_107 = torch.tensor_split(v_105, dim=1, indices=(116,))
        v_108 = self.onnx__Conv_1321_data
        v_109 = self.onnx__Conv_1322_data
        v_110 = Conv(v_107, v_108, v_109, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_111 = F.relu(v_110)
        v_112 = self.onnx__Conv_1324_data
        v_113 = self.onnx__Conv_1325_data
        v_114 = Conv(v_111, v_112, v_113, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_115 = self.onnx__Conv_1327_data
        v_116 = self.onnx__Conv_1328_data
        v_117 = Conv(v_114, v_115, v_116, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_118 = F.relu(v_117)
        v_119 = torch.cat((v_106, v_118), dim=1)
        v_120 = self.channelshuffle_6(v_119)
        v_121, v_122 = torch.tensor_split(v_120, dim=1, indices=(116,))
        v_123 = self.onnx__Conv_1330_data
        v_124 = self.onnx__Conv_1331_data
        v_125 = Conv(v_122, v_123, v_124, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_126 = F.relu(v_125)
        v_127 = self.onnx__Conv_1333_data
        v_128 = self.onnx__Conv_1334_data
        v_129 = Conv(v_126, v_127, v_128, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_130 = self.onnx__Conv_1336_data
        v_131 = self.onnx__Conv_1337_data
        v_132 = Conv(v_129, v_130, v_131, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_133 = F.relu(v_132)
        v_134 = torch.cat((v_121, v_133), dim=1)
        v_135 = self.channelshuffle_7(v_134)
        v_136, v_137 = torch.tensor_split(v_135, dim=1, indices=(116,))
        v_138 = self.onnx__Conv_1339_data
        v_139 = self.onnx__Conv_1340_data
        v_140 = Conv(v_137, v_138, v_139, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_141 = F.relu(v_140)
        v_142 = self.onnx__Conv_1342_data
        v_143 = self.onnx__Conv_1343_data
        v_144 = Conv(v_141, v_142, v_143, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_145 = self.onnx__Conv_1345_data
        v_146 = self.onnx__Conv_1346_data
        v_147 = Conv(v_144, v_145, v_146, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_148 = F.relu(v_147)
        v_149 = torch.cat((v_136, v_148), dim=1)
        v_150 = self.channelshuffle_8(v_149)
        v_151, v_152 = torch.tensor_split(v_150, dim=1, indices=(116,))
        v_153 = self.onnx__Conv_1348_data
        v_154 = self.onnx__Conv_1349_data
        v_155 = Conv(v_152, v_153, v_154, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_156 = F.relu(v_155)
        v_157 = self.onnx__Conv_1351_data
        v_158 = self.onnx__Conv_1352_data
        v_159 = Conv(v_156, v_157, v_158, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_160 = self.onnx__Conv_1354_data
        v_161 = self.onnx__Conv_1355_data
        v_162 = Conv(v_159, v_160, v_161, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_163 = F.relu(v_162)
        v_164 = torch.cat((v_151, v_163), dim=1)
        v_165 = self.channelshuffle_9(v_164)
        v_166, v_167 = torch.tensor_split(v_165, dim=1, indices=(116,))
        v_168 = self.onnx__Conv_1357_data
        v_169 = self.onnx__Conv_1358_data
        v_170 = Conv(v_167, v_168, v_169, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_171 = F.relu(v_170)
        v_172 = self.onnx__Conv_1360_data
        v_173 = self.onnx__Conv_1361_data
        v_174 = Conv(v_171, v_172, v_173, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_175 = self.onnx__Conv_1363_data
        v_176 = self.onnx__Conv_1364_data
        v_177 = Conv(v_174, v_175, v_176, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_178 = F.relu(v_177)
        v_179 = torch.cat((v_166, v_178), dim=1)
        v_180 = self.channelshuffle_10(v_179)
        v_181, v_182 = torch.tensor_split(v_180, dim=1, indices=(116,))
        v_183 = self.onnx__Conv_1366_data
        v_184 = self.onnx__Conv_1367_data
        v_185 = Conv(v_182, v_183, v_184, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_186 = F.relu(v_185)
        v_187 = self.onnx__Conv_1369_data
        v_188 = self.onnx__Conv_1370_data
        v_189 = Conv(v_186, v_187, v_188, dilations=(1,1), group=116, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_190 = self.onnx__Conv_1372_data
        v_191 = self.onnx__Conv_1373_data
        v_192 = Conv(v_189, v_190, v_191, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_193 = F.relu(v_192)
        v_194 = torch.cat((v_181, v_193), dim=1)
        v_195 = self.channelshuffle_11(v_194)
        v_196 = self.onnx__Conv_1375_data
        v_197 = self.onnx__Conv_1376_data
        v_198 = Conv(v_195, v_196, v_197, dilations=(1,1), group=232, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_199 = self.onnx__Conv_1378_data
        v_200 = self.onnx__Conv_1379_data
        v_201 = Conv(v_198, v_199, v_200, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_202 = F.relu(v_201)
        v_203 = self.onnx__Conv_1381_data
        v_204 = self.onnx__Conv_1382_data
        v_205 = Conv(v_195, v_203, v_204, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_206 = F.relu(v_205)
        v_207 = self.onnx__Conv_1384_data
        v_208 = self.onnx__Conv_1385_data
        v_209 = Conv(v_206, v_207, v_208, dilations=(1,1), group=232, kernel_shape=(3,3), pads=(1,1,1,1), strides=(2,2))
        v_210 = self.onnx__Conv_1387_data
        v_211 = self.onnx__Conv_1388_data
        v_212 = Conv(v_209, v_210, v_211, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_213 = F.relu(v_212)
        v_214 = torch.cat((v_202, v_213), dim=1)
        v_215 = self.channelshuffle_12(v_214)
        v_216, v_217 = torch.tensor_split(v_215, dim=1, indices=(232,))
        v_218 = self.onnx__Conv_1390_data
        v_219 = self.onnx__Conv_1391_data
        v_220 = Conv(v_217, v_218, v_219, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_221 = F.relu(v_220)
        v_222 = self.onnx__Conv_1393_data
        v_223 = self.onnx__Conv_1394_data
        v_224 = Conv(v_221, v_222, v_223, dilations=(1,1), group=232, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_225 = self.onnx__Conv_1396_data
        v_226 = self.onnx__Conv_1397_data
        v_227 = Conv(v_224, v_225, v_226, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_228 = F.relu(v_227)
        v_229 = torch.cat((v_216, v_228), dim=1)
        v_230 = self.channelshuffle_13(v_229)
        v_231, v_232 = torch.tensor_split(v_230, dim=1, indices=(232,))
        v_233 = self.onnx__Conv_1399_data
        v_234 = self.onnx__Conv_1400_data
        v_235 = Conv(v_232, v_233, v_234, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_236 = F.relu(v_235)
        v_237 = self.onnx__Conv_1402_data
        v_238 = self.onnx__Conv_1403_data
        v_239 = Conv(v_236, v_237, v_238, dilations=(1,1), group=232, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_240 = self.onnx__Conv_1405_data
        v_241 = self.onnx__Conv_1406_data
        v_242 = Conv(v_239, v_240, v_241, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_243 = F.relu(v_242)
        v_244 = torch.cat((v_231, v_243), dim=1)
        v_245 = self.channelshuffle_14(v_244)
        v_246, v_247 = torch.tensor_split(v_245, dim=1, indices=(232,))
        v_248 = self.onnx__Conv_1408_data
        v_249 = self.onnx__Conv_1409_data
        v_250 = Conv(v_247, v_248, v_249, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_251 = F.relu(v_250)
        v_252 = self.onnx__Conv_1411_data
        v_253 = self.onnx__Conv_1412_data
        v_254 = Conv(v_251, v_252, v_253, dilations=(1,1), group=232, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_255 = self.onnx__Conv_1414_data
        v_256 = self.onnx__Conv_1415_data
        v_257 = Conv(v_254, v_255, v_256, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_258 = F.relu(v_257)
        v_259 = torch.cat((v_246, v_258), dim=1)
        v_260 = self.channelshuffle_15(v_259)
        v_261 = self.lateral_convs_2_weight_data
        v_262 = self.lateral_convs_2_bias_data
        v_263 = Conv(v_260, v_261, v_262, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_264 = self.lateral_convs_1_weight_data
        v_265 = self.lateral_convs_1_bias_data
        v_266 = Conv(v_195, v_264, v_265, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_267 = self.lateral_convs_0_weight_data
        v_268 = self.lateral_convs_0_bias_data
        v_269 = Conv(v_70, v_267, v_268, dilations=(1,1), group=1, kernel_shape=(1,1), pads=(0,0,0,0), strides=(1,1))
        v_270 = self.onnx__Conv_1417_data
        v_271 = self.onnx__Conv_1418_data
        v_272 = Conv(v_263, v_270, v_271, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_273 = F.relu(v_272)
        v_274 = self.onnx__Conv_1420_data
        v_275 = self.onnx__Conv_1421_data
        v_276 = Conv(v_273, v_274, v_275, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_277 = F.relu(v_276)
        v_278 = self.onnx__Conv_1423_data
        v_279 = self.onnx__Conv_1424_data
        v_280 = Conv(v_263, v_278, v_279, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_281 = F.relu(v_280)
        v_282 = self.onnx__Conv_1426_data
        v_283 = self.onnx__Conv_1427_data
        v_284 = Conv(v_281, v_282, v_283, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_285 = F.relu(v_284)
        v_286 = self.gfl_cls_weight_data
        v_287 = self.gfl_cls_bias_data
        v_288 = Conv(v_277, v_286, v_287, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_289 = self.gfl_reg_weight_data
        v_290 = self.gfl_reg_bias_data
        v_291 = Conv(v_285, v_289, v_290, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_292 = torch.exp(v_291)
        v_293 = v_263.to(copy=False, dtype=torch.float, memory_format=torch.preserve_format, non_blocking=False)
        v_294 = F.interpolate(v_293, mode='nearest', recompute_scale_factor=False, scale_factor=(2.0,2.0))
        v_295 = v_294.to(copy=False, dtype=torch.half, memory_format=torch.preserve_format, non_blocking=False)
        v_296 = (v_266 + v_295)
        v_297 = Conv(v_296, v_270, v_271, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_298 = F.relu(v_297)
        v_299 = Conv(v_298, v_274, v_275, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_300 = F.relu(v_299)
        v_301 = Conv(v_296, v_278, v_279, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_302 = F.relu(v_301)
        v_303 = Conv(v_302, v_282, v_283, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_304 = F.relu(v_303)
        v_305 = Conv(v_300, v_286, v_287, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_306 = Conv(v_304, v_289, v_290, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_307 = torch.exp(v_306)
        v_308 = v_296.to(copy=False, dtype=torch.float, memory_format=torch.preserve_format, non_blocking=False)
        v_309 = F.interpolate(v_308, mode='nearest', recompute_scale_factor=False, scale_factor=(2.0,2.0))
        v_310 = v_309.to(copy=False, dtype=torch.half, memory_format=torch.preserve_format, non_blocking=False)
        v_311 = (v_269 + v_310)
        v_312 = Conv(v_311, v_270, v_271, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_313 = F.relu(v_312)
        v_314 = Conv(v_313, v_274, v_275, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_315 = F.relu(v_314)
        v_316 = Conv(v_311, v_278, v_279, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_317 = F.relu(v_316)
        v_318 = Conv(v_317, v_282, v_283, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_319 = F.relu(v_318)
        v_320 = Conv(v_315, v_286, v_287, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_321 = Conv(v_319, v_289, v_290, dilations=(1,1), group=1, kernel_shape=(3,3), pads=(1,1,1,1), strides=(1,1))
        v_322 = torch.exp(v_321)
        return v_320, v_305, v_288, v_322, v_307, v_292

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.half)

    mod = torch.jit.trace(net, v_0)
    mod.save("nanodet_320_fp16_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.half)

    torch.onnx.export(net, v_0, "nanodet_320_fp16_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0', 'out1', 'out2', 'out3', 'out4', 'out5'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.half)

    import pnnx
    pnnx.export(net, "nanodet_320_fp16_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.half)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
