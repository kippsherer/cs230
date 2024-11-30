
#statistics = ['binary_crossentropy', 'accuracy', 'precision', 'recall', 'f1_score', 'false_negatives', 'false_positives']

model_statistics = {
    'custom 1a':{
        'training':{'accuracy': [0.9998157620429993, 0.9993183016777039, 1.0, 1.0, 1.0, 1.0, 1.0], 'f1_score': [0.934715986251831, 0.8912848830223083, 0.9271858930587769, 0.9510853290557861, 0.9582425951957703, 0.965788722038269, 0.9714637398719788], 'false_negatives_1': [5.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'false_positives_1': [5.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'loss': [0.0008897430379875004, 0.002739389194175601, 4.6128188841976225e-05, 8.9712739281822e-06, 7.772139724693261e-06, 7.197891136456747e-06, 6.833802217443008e-06], 'precision_1': [0.9996657967567444, 0.9989302754402161, 1.0, 1.0, 1.0, 1.0, 1.0], 'recall_1': [0.9996657967567444, 0.9985964298248291, 1.0, 1.0, 1.0, 1.0, 1.0]},
        'dev':{'binary_crossentropy': 0.03696813806891441, 'accuracy': 0.9963333606719971, 'precision': 0.991062581539154, 'recall': 0.9980000257492065, 'f1_score': 0.9694619178771973, 'false_negatives': 2.0, 'false_positives': 9.0},
        'test':{'binary_crossentropy': 0.006234058178961277, 'accuracy': 0.9983333349227905, 'precision': 0.9989969730377197, 'recall': 0.9959999918937683, 'f1_score': 0.9727625250816345, 'false_negatives': 4.0, 'false_positives': 1.0},
        'test_dark':{'binary_crossentropy': 0.04195870831608772, 'accuracy': 0.9868420958518982, 'precision': 1.0, 'recall': 0.9866071343421936, 'f1_score': 1.0, 'false_negatives': 3.0, 'false_positives': 0.0},
    },
    'MobileNetV2 1a':{
        'training':{'accuracy': [0.9470280408859253, 0.9725098609924316, 0.9768766164779663, 0.9800456762313843, 0.9809300899505615, 0.9815749526023865, 0.9825515151023865, 0.9831042289733887, 0.9830121397972107, 0.9835464358329773], 'f1_score': [0.43220284581184387, 0.43239644169807434, 0.4330848455429077, 0.4350113868713379, 0.4384147524833679, 0.4422701299190521, 0.4453771114349365, 0.4487029016017914, 0.451567143201828, 0.45649945735931396], 'false_negatives': [1600.0, 703.0, 588.0, 502.0, 461.0, 449.0, 432.0, 415.0, 414.0, 405.0], 'false_positives': [1275.0, 789.0, 667.0, 581.0, 574.0, 551.0, 515.0, 502.0, 508.0, 488.0], 'loss': [0.141740620136261, 0.07953828573226929, 0.06653671711683273, 0.05872257798910141, 0.05635808780789375, 0.05333400145173073, 0.05009522661566734, 0.04902678728103638, 0.047765910625457764, 0.04645762965083122], 'precision': [0.9128919839859009, 0.9475677609443665, 0.9556545615196228, 0.9613722562789917, 0.9619237184524536, 0.9634227156639099, 0.965769350528717, 0.9666423201560974, 0.9662593007087708, 0.9675639867782593], 'recall': [0.8930624127388, 0.9530143141746521, 0.9607004523277283, 0.9664483070373535, 0.9691886305809021, 0.9699906706809998, 0.9711268544197083, 0.9722630381584167, 0.9723299145698547, 0.972931444644928]},
        'dev':{'binary_crossentropy': 0.03182130306959152, 'accuracy': 0.9879999756813049, 'precision': 0.9772277474403381, 'recall': 0.9869999885559082, 'f1_score': 0.5219206213951111, 'false_negatives': 13.0, 'false_positives': 23.0},
        'test':{'binary_crossentropy': 0.028676219284534454, 'accuracy': 0.9926666617393494, 'precision': 0.9890000224113464, 'recall': 0.9890000224113464, 'f1_score': 0.5263157486915588, 'false_negatives': 11.0, 'false_positives': 11.0},
        'test_dark':{'binary_crossentropy': 0.07305243611335754, 'accuracy': 0.9649122953414917, 'precision': 1.0, 'recall': 0.9642857313156128, 'f1_score': 0.9955555200576782, 'false_negatives': 8.0, 'false_positives': 0.0},
    },
    'MobileNetV2 1b lr10x':{
        'training':{},
        'dev':{},
        'test':{},
        'test_dark':{},
    },




    'custom Nx':{
        'training':{},
        'dev':{},
        'test':{},
        'test_dark':{},
    },


}

