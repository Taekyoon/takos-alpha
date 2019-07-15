def process_by_ner(text, tags):
    tag_stack = []
    entity_stack = []

    entities = []
    entity_tags = []

    for n, t in zip(tags, text):
        if not (n == 'O' or n == '<pad>'):
            state, tag = n.split('-')
            if state == 'B':
                if len(entity_stack) > 0:
                    entities.append(' '.join(entity_stack))
                    entity_tags.append(tag_stack[0])
                    entity_stack = []
                    tag_stack = []

                entity_stack.append(t)
                tag_stack.append(tag)

            elif state == 'I':
                if len(entity_stack) > 0:
                    entity_stack.append(t)
                    tag_stack.append(tag)

        else:
            if len(entity_stack) > 0:
                entities.append(' '.join(entity_stack))
                entity_tags.append(tag_stack[0])
                entity_stack = []
                tag_stack = []

    if len(entity_stack) > 0:
        entities.append(' '.join(entity_stack))
        entity_tags.append(tag_stack[0])

    return entity_tags, entities
