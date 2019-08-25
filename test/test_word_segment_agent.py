from takos.agents.word_segment import WordSegmentAgent


def test_build_word_segment():
    config_path = './test/scripts/bilstm_configs.json'
    agent = WordSegmentAgent(config_path)

    assert isinstance(agent, object)


def test_train_word_segment():
    config_path = './test/scripts/bilstm_configs.json'
    agent = WordSegmentAgent(config_path)

    agent.train()

    assert True


def test_eval_word_segment():
    config_path = './test/scripts/bilstm_configs.json'
    agent = WordSegmentAgent(config_path)

    agent.eval()

    assert True


def test_run_word_segment():
    config_path = './test/scripts/bilstm_configs.json'
    agent = WordSegmentAgent(config_path)
    test_text = '동해물과백두산이'

    segmented_text = agent(test_text)

    assert isinstance(segmented_text, dict)
