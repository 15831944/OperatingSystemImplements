#include <pthread.h>
#include <curses.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define LOG_NUM       15		//lines of rivers
#define WOOD_NUM       5		//wood per line
#define LINE_LENGTH  161        //Output total length
#define GOAL_ROW   1            //First row on display : GOAL!
#define START_ROW  GOAL_ROW + LOG_NUM + 1 // Starting row on display : Start
/*State defining*/
#define PLAYING 999
#define WIN     1000
#define DIE     998
#define QUIT    997
#define LEFT    1
#define RIGHT   2
#define UP      3
#define DOWN    4

typedef enum { FROG, LOG } Type;
typedef struct
{
    int id;
    int dir;
    int row;
    long floatduration;
} Log;

int state = 0;                                              // Frog state!
Log logs[LOG_NUM];                                          // Global Logs
char lines[LOG_NUM+2][LINE_LENGTH];                         // lines on window with start && goal
pthread_mutex_t display_lock = PTHREAD_MUTEX_INITIALIZER;   // pthread display lock (needed

static int frog_row = START_ROW, frog_col = (WOOD_NUM * 4);
char under_frog = '|';
const char frog = 'Y';

float mixing(float m)
{
     return m*rand()/(RAND_MAX);
}

void init_rivers (int nol, int lpl)
{
    int i = 0, j;
    const char passage[] = "||||||||";
    const char floatingwood[] = "====    ";
    while (i++ < lpl)
    {
	strcat (lines[GOAL_ROW - 1], passage);
	strcat (lines[START_ROW - GOAL_ROW], passage);
	for (j = 1 ; j <= nol ; j++)
	    strcat (lines[j], floatingwood);
    }
}

inline void drawline (char *line, int row)
{
    move (row, 0);
    printw (line);
}   /*thanks curses!*/

void rotate_river (char *theline, int dir) {
    char tmpline[LINE_LENGTH];
    int len;

    len = strlen (theline);
    strcpy (tmpline, theline);
    if (dir == LEFT)
    {
	strncpy (theline, tmpline+1, len-1);
	theline[len-1] = tmpline[0];
    }
    else
    {
	strncpy (theline+1, tmpline, len-1);
	theline[0] = tmpline[len-1];
    }
}


void DrawFrog()
{
    move(frog_row, frog_col);
    printw("%c",frog);
}
void FrogWake()
{
    move(frog_row, frog_col);
    printw("%c",under_frog);
}
void MoveFrog(int state)
{
    FrogWake();
    switch(state)
    {
    case UP:    frog_row--; break;
    case DOWN:  frog_row++; break;
    case LEFT:  frog_col--; break;
    case RIGHT: frog_col++; break;
    default: break;
    }
    DrawFrog();
}

int LOGDIR(int dir)
{
    return (dir == LEFT)? -1:1;
}
void MoveLog(int row,int dir)
{
    rotate_river(lines[(row)-GOAL_ROW], dir);
    drawline(lines[row-GOAL_ROW], row);
    if(frog_row == row)
        frog_col += LOGDIR(dir);

    DrawFrog();
}
int OutOfBound(int row, int col)
{
    if(row > START_ROW || row < GOAL_ROW || col < 0 || col > (strlen(lines[0])-1))
        return 1;
    else
        return 0;
}

void display (Type who, int d) {

    pthread_mutex_lock (&display_lock);
    if (who == LOG)
	{
	    MoveLog (logs[d].row, logs[d].dir);
    }
    else
	switch (d) {
	  case UP:
	      if (frog_row != GOAL_ROW) { MoveFrog (d); }
	      else state = WIN;
	      break;
	  case DOWN:
	      if (frog_row != START_ROW) { MoveFrog (d); }  break;
	  case LEFT:
	      if (frog_col != 0) { MoveFrog (d); }  break;
	  case RIGHT:
	      if (frog_col != strlen (lines[0])-1) { MoveFrog (d); }  break;
	  default:
	      printf ("Error in display state=%d\n", d); break;
	}

    under_frog = lines[frog_row - GOAL_ROW][frog_col];
    if ((under_frog == ' ') || OutOfBound (frog_row, frog_col))
	state = DIE;

    move (0,0);
    refresh ();
    pthread_mutex_unlock (&display_lock);
}

Log summon_log (int id, int dir, int row, long floatduration) {
    Log log;
    log.id = id;
    log.dir = dir;
    log.row = row;
    log.floatduration = floatduration;
    return log;
}

void *logf (void* tlog) {
    Log *log;
    log = (Log *)tlog;
    while (state == PLAYING)
    {
	display (LOG, log->id);
    usleep (log->floatduration);  //microsecond sleep
    }
    return NULL;
}

int main () {
    pthread_t log_thread[LOG_NUM];
    int i;
    int dir = RIGHT;                // first pile go left
    char command;

    printf("Please input the difficulty you want (1-10 lower is harder)\n");
    int x = 0;
    scanf("%d",&x);
    init_rivers (LOG_NUM, WOOD_NUM);
    for (i = 0 ; i < LOG_NUM ; i++) {
	logs[i] = summon_log (i, dir, (GOAL_ROW + i + 1), mixing((float)x*0.1)*1000000 ); // floattime from 0~0.5sec trans into m
	if (dir == LEFT)
        dir = RIGHT;
    else
        dir = LEFT;
    }

    state = PLAYING;

    initscr ();
    clear ();
    cbreak ();
    noecho ();
    timeout (100);

    for (i = 0 ; i < LOG_NUM ; i++)
	pthread_create (&log_thread[i], NULL, &logf, &logs[i]);

    drawline (lines[START_ROW - GOAL_ROW], START_ROW);
    drawline (lines[GOAL_ROW - GOAL_ROW],  GOAL_ROW);
    display (FROG, DOWN);

    while (state == PLAYING) {
	command = getch ();
	switch (command) {
	  case 'a':
	      display (FROG, LEFT);
	      break;
	  case 'd':
	      display (FROG, RIGHT);
	      break;
	  case 'w':
	      display (FROG, UP);
	      break;
	  case 's':
	      display (FROG, DOWN);
	      break;
	  case 'q':
	      state = QUIT;
	      break;
	  default:   break;
	}
    }

    for (i = 0 ; i < LOG_NUM ; i++)
	pthread_join (log_thread[0], NULL);
    /* Terminate Threads */

    echo ();
    nocbreak ();
    endwin ();

    switch (state) {
      case WIN:
	  printf ("Mr.Froggy survived from hell!\n");  break;
      case DIE:
	  printf ("Rest in peace Mr.Froggy!\n");  break;
      case QUIT:
	  printf ("ByeBye Zzz\n");  break;
      default:
	  printf ("Err State = %d\n", state);
    }

    exit (0);
}
