
import logging

logger = logging.getLogger(__name__)


class EarlyStopping(object):

    def __init__(self, patience, delta=0, high_record=False):
        """
        Initialize an EarlyStopping instance
        Args:
            patience: How long to wait after last time validation loss decreased
            delta: Minimum change in the monitored quantity to qualify as an improvement
            high_record: True if the improvement of the record is seen as the improvement of the performance,
                default False
        """
        """

        :param patience: 
        :param delta: 
        :param high_record: 
        """
        self.patience = patience
        self.counter = 0
        self.record = None
        self.early_stop = False
        self.delta = delta
        self.high_record = high_record

        self.refreshed = False
        self.best_model = None
        self.best_epoch = -1

    def __call__(self, score, model, epoch):
        """
        Call this instance when get a new score
        Args:
            score (float): the new score
            model:
        """
        # first call
        if self.record is None:
            self.record = score
            self.refreshed = True
            self.best_model = model
            self.best_epoch = epoch
        # not hit the best
        elif (not self.high_record and score > self.record + self.delta) or \
                (self.high_record and score < self.record - self.delta):
            self.counter += 1
            self.refreshed = False
            logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning('Early stop')
        # hit the best
        else:
            self.record = score
            self.counter = 0
            self.refreshed = True
            self.best_model = model
            self.best_epoch = epoch
