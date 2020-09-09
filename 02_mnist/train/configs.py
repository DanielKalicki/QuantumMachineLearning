import copy

def gen_name(config):
    network_type = config['network_type'][0]
    classes = str(config['classes']).replace(" ", "")
    h = str(config['image_h'])
    w = str(config['image_w'])
    if (config['network_type'] == 'classic') and (config['classic']['linear']):
        network_type += '_linear'
    if (config['network_type'] == 'quantum') and (config['quantum']['layers']):
        network_type += '_' + str(config['quantum']['layers'])+'l'

    return network_type + "_" + h + "x" + w + "_" + classes + "_"

default_config = {
    'batch_size': 1,
    'image_h': 3,
    'image_w': 3,
    'network_type': 'quantum',
    'classic': {
        'linear': False
    },
    'quantum': {
        'layers': 1
    },
    'classes': [0, 1, 2, 3, 4],
    # 'name': 'circuit02_3x3_5class_AncLayersxU3.CNOT_5AncilaClass_1000shots_CEloss_gpuTest'
    'name': ''
}

configs = []
for i in range(1000):
    configs.append(copy.deepcopy(default_config))

test_classes = [
    [6, 7], [6, 9],
    [5, 8], [3, 5],
    [4, 8], [4, 7], [0, 4]
]

i = 0
for class_ in test_classes:
    configs[i]['image_h'] = 2
    configs[i]['image_w'] = 2
    configs[i]['network_type'] = 'classic'
    configs[i]['classes'] = class_
    configs[i]['name'] = gen_name(configs[i]) + str(i) +'v6'
    i += 1

# i = 7
for class_ in test_classes:
    configs[i]['image_h'] = 2
    configs[i]['image_w'] = 2
    configs[i]['network_type'] = 'quantum'
    configs[i]['classes'] = class_
    configs[i]['name'] = gen_name(configs[i])+ "1xArbU_" + str(i) +'r4'
    i += 1

# i = 14
for class_ in test_classes:
    configs[i]['image_h'] = 2
    configs[i]['image_w'] = 2
    configs[i]['network_type'] = 'quantum'
    configs[i]['quantum']['layers'] = 2
    configs[i]['classes'] = class_
    configs[i]['name'] = gen_name(configs[i])+ "2xArbU_" + str(i) +'r1'
    i += 1

# i = 0
# for c1 in range(9):
#     for c2 in range(c1+1, 10):
#         configs[i]['image_h'] = 2
#         configs[i]['image_w'] = 2
#         configs[i]['network_type'] = 'classic'
#         configs[i]['classes'] = [c1, c2]
#         configs[i]['name'] = gen_name(configs[i]) + str(i)
#         i += 1

# # i = 45
# for c1 in range(9):
#     for c2 in range(c1+1, 10):
#         configs[i]['image_h'] = 2
#         configs[i]['image_w'] = 2
#         configs[i]['network_type'] = 'quantum'
#         configs[i]['classes'] = [c1, c2]
#         configs[i]['name'] = gen_name(configs[i])+ "1xArbU_" + str(i)
#         i += 1

# # i = 90
# for c1 in range(9):
#     for c2 in range(c1+1, 10):
#         configs[i]['image_h'] = 2
#         configs[i]['image_w'] = 2
#         configs[i]['network_type'] = 'quantum'
#         configs[i]['quantum']['layers'] = 2
#         configs[i]['classes'] = [c1, c2]
#         configs[i]['name'] = gen_name(configs[i])+ "2xArbU" + str(i)
#         i += 1

# # i = 135
# for c1 in range(9):
#     for c2 in range(c1+1, 10):
#         configs[i]['image_h'] = 2
#         configs[i]['image_w'] = 2
#         configs[i]['network_type'] = 'quantum'
#         configs[i]['quantum']['layers'] = 3
#         configs[i]['classes'] = [c1, c2]
#         configs[i]['name'] = gen_name(configs[i])+ "3xArbU" + str(i)
#         i += 1