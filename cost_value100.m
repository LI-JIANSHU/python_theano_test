function [cost] = cost_value100(input,phi,n_s,r_phi)

% cost function with reduce matrix A and E
% used in angle_solver4

input=input(:);
theta2=input(1:n_s);
P=input(n_s+1:end);

n_phi=length(phi);
m2=length(r_phi);
n_a=sqrt(m2/n_phi);

G=zeros(n_a^2,2*n_a-1);
for k=1:n_a
    G(((k-1)*n_a+1):(k*n_a),(n_a-k+1):(2*n_a-k))=eye(n_a);
end

B_phi=zeros(n_a^2*n_phi,n_s);
for k=1:n_phi

    B=zeros(2*n_a-1,n_s);
    for k2=1:n_s
    B(:,k2)=exp([-(n_a-1):1:(n_a-1)]'.*1i*pi*sin((theta2(k2)+phi(k))*pi/180));
    end

    start=(k-1)*n_a^2;
    B_phi(start+1:start+n_a^2,:)=G*B;

end

projector=zeros(m2);
part1=eye(n_a^2)*(n_phi-1)/n_phi;
part2=-eye(n_a^2)/n_phi;
for row=1:n_phi
	for col=1:n_phi
		row_start=(row-1)*n_a^2;
		col_start=(col-1)*n_a^2;
		if row==col
			projector(row_start+1:row_start+n_a^2,col_start+1:col_start+n_a^2)=part1;
		else
			projector(row_start+1:row_start+n_a^2,col_start+1:col_start+n_a^2)=part2;
		end
	end
end
%projector=eye(m2);

temp=projector*B_phi*P-projector*r_phi;
cost=norm(temp(:));
% cost=max(abs(temp(:)));

grad=zeros(size(input));
for k=1:n_s
    
end
end