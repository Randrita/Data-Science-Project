# include<stdio.h>
# include<stdlib.h>
typedef struct List
	{
	float num;
	struct List* next;
	}Node;

Node* insertbeg(Node *head)
	{
	Node *new;
	new=(Node*)malloc(sizeof(Node));
	printf("\nEnter data : ");
	scanf("%f",&new->num);
	new->next=head;
	head=new;
	return head;
	}

Node* insertend(Node *head)
	{
	Node *ptr,*new;
	new=(Node*)malloc(sizeof(Node));
	printf("\nEnter data : ");
	scanf("%f",&new->num);
	ptr=head;
	if(head==NULL)
		{
		head=new;
		new->next=NULL;
		return head;
		}
	while(ptr->next!=NULL)
		{
		ptr=ptr->next;
		}
	ptr->next=new;
	new->next=NULL;
	return head;
	}

Node* insertafter(Node *head,float f)
	{
	Node *ptr,*new;
	int p=0;
	new=(Node*)malloc(sizeof(Node));
	printf("\nEnter data : ");
	scanf("%f",&new->num);
	ptr=head;
	if(head==NULL)
		{
		printf("\nInsertion failed!!!\n%f not found , Linked List empty.",f);
		return head;
		}
	while(ptr!=NULL)
		{
		if(ptr->num==f)
			{
			p=1;
			new->next=ptr->next;
			ptr->next=new;
			}
		ptr=ptr->next;
		}
	if(p!=1)
	printf("\nInsertion failed!!!\n%f not found in the Linked List",f);
	return head;
	}

void display(Node *head)
	{
	Node *ptr;
	ptr=head;
	if(head==NULL)
		{
		printf("\nLinked List empty!!");
		return;
		}
	printf("\n| ");
	while(ptr!=NULL)
		{
		printf("%f | ",ptr->num);
		ptr=ptr->next;
		}
	}

void main()
	{
	int ch;
	float f;
	Node *head;
	head=NULL;
	
	while(1)
	{
	printf("\n1. Insert at beginning");
	printf("\n2. Insert at end");
	printf("\n3. Insert after a node");
	
	printf("\n4. display");
	printf("\n5. exit");
	printf("\nEnter your choice : ");
	scanf("%d",&ch);
	switch(ch)
		{
		case 1:
		head=insertbeg(head);
		break;
		
		case 2:
		head=insertend(head);
		break;
		
		case 3:
		printf("\nEnter the number after whose node u want to enter : ");
		scanf("%f",&f);
		head=insertafter(head,f);
		break;
		
		case 4:
		display(head);
		break;
		
		case 5:
		if(head!=NULL)
			while(head!=NULL)
				{
				Node *ptr=head;
				free (ptr);
				head=head->next;
				}
		display(head);
		printf("\n");
			
		exit(0);
		
		default:
		printf("\nWrong choice !!!");
		}
	}
	}
