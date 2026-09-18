"""Run Rasa's NLU evaluation and write the report files."""

import asyncio

from rasa.model_testing import test_nlu

from .get_model import get_model


def evaluate(model_path, nlu_data, output_dir):
    """Evaluate a trained model and persist the reports under ``output_dir``.

    Rasa 3.x names the plots itself -- ``intent_confusion_matrix.png``,
    ``intent_histogram.png`` and the ``DIETClassifier_*`` equivalents -- along
    with the per-intent/entity JSON reports, all inside ``output_dir``. The
    ``confmat``/``histogram`` filename options of Rasa 1.x are gone.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(test_nlu(
        model=get_model(model_path),
        nlu_data=str(nlu_data),
        output_directory=str(output_dir),
        additional_arguments={'successes': True, 'errors': True},
    ))
    print('Evaluation reports written to {}'.format(output_dir))
